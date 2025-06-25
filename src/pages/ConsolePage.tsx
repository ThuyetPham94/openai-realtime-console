/**
 * Voice Chat Console Page integrated with WebSocket server
 * Connects to local sockets.py server at ws://localhost:8000/ws
 */

import { useEffect, useRef, useCallback, useState } from 'react';
import { WavRecorder, WavStreamPlayer } from '../lib/wavtools/index.js';
import { WavRenderer } from '../utils/wav_renderer';

import { X, Edit, Zap, ArrowUp, ArrowDown, Mic, MicOff, Volume2, VolumeX } from 'react-feather';
import { Button } from '../components/button/Button';
import { Toggle } from '../components/toggle/Toggle';

import './ConsolePage.scss';

/**
 * Type for WebSocket messages
 */
interface WebSocketMessage {
  type: string;
  [key: string]: any;
}

/**
 * Type for conversation items
 */
interface ConversationItem {
  id: string;
  timestamp: string;
  type: 'user' | 'assistant' | 'system';
  transcript?: string;
  response?: string;
  audio_url?: string;
}

/**
 * Type for system logs
 */
interface SystemLog {
  id: string;
  timestamp: string;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
  source: 'client' | 'server' | 'system';
}

export function ConsolePage() {
  // WebSocket and connection state
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [sessionId, setSessionId] = useState('');
  const [serverUrl, setServerUrl] = useState('ws://localhost:8000/ws');
  
  // Audio and processing state
  const [audioLevel, setAudioLevel] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [useManualMode, setUseManualMode] = useState(true);
  
  // Data state
  const [conversationItems, setConversationItems] = useState<ConversationItem[]>([]);
  const [systemLogs, setSystemLogs] = useState<SystemLog[]>([]);
  const [serverInfo, setServerInfo] = useState<any>(null);

  // Refs
  const wsRef = useRef<WebSocket | null>(null);
  const wavRecorderRef = useRef<WavRecorder>(new WavRecorder({ sampleRate: 16000 }));
  const wavStreamPlayerRef = useRef<WavStreamPlayer>(new WavStreamPlayer({ sampleRate: 16000 }));
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  
  // Canvas refs for visualization
  const clientCanvasRef = useRef<HTMLCanvasElement>(null);
  const serverCanvasRef = useRef<HTMLCanvasElement>(null);
  
  // Scroll refs
  const eventsScrollRef = useRef<HTMLDivElement>(null);
  const eventsScrollHeightRef = useRef(0);
  const startTimeRef = useRef<string>(new Date().toISOString());

  /**
   * Utility functions
   */
  const formatTime = useCallback((timestamp: string) => {
    const startTime = startTimeRef.current;
    const t0 = new Date(startTime).valueOf();
    const t1 = new Date(timestamp).valueOf();
    const delta = t1 - t0;
    const hs = Math.floor(delta / 10) % 100;
    const s = Math.floor(delta / 1000) % 60;
    const m = Math.floor(delta / 60_000) % 60;
    const pad = (n: number) => {
      let s = n + '';
      while (s.length < 2) {
        s = '0' + s;
      }
      return s;
    };
    return `${pad(m)}:${pad(s)}.${pad(hs)}`;
  }, []);

  const addLog = useCallback((message: string, type: 'info' | 'success' | 'warning' | 'error' = 'info', source: 'client' | 'server' | 'system' = 'system') => {
    const log: SystemLog = {
      id: Date.now().toString(),
      timestamp: new Date().toISOString(),
      message,
      type,
      source
    };
    setSystemLogs(prev => [...prev.slice(-49), log]);
  }, []);

  const addConversationItem = useCallback((transcript: string, response: string, audio_url?: string) => {
    const item: ConversationItem = {
      id: Date.now().toString(),
      timestamp: new Date().toISOString(),
      type: 'user',
      transcript,
      response,
      audio_url
    };
    setConversationItems(prev => [...prev.slice(-19), item]);
  }, []);

  /**
   * Audio initialization and management
   */
  const initializeAudio = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });

      streamRef.current = stream;
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: 16000
      });

      const analyser = audioContextRef.current.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.8;
      analyserRef.current = analyser;

      const source = audioContextRef.current.createMediaStreamSource(stream);
      source.connect(analyser);

      addLog('🎤 Microphone initialized successfully', 'success', 'client');
      return true;
    } catch (error: any) {
      addLog(`❌ Microphone error: ${error.message}`, 'error', 'client');
      return false;
    }
  }, [addLog]);

  const updateAudioLevel = useCallback(() => {
    if (!analyserRef.current) return;

    const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
    analyserRef.current.getByteFrequencyData(dataArray);

    const average = dataArray.reduce((sum, value) => sum + value, 0) / dataArray.length;
    setAudioLevel(average / 255);

    // Draw visualization
    if (clientCanvasRef.current && isRecording) {
      const canvas = clientCanvasRef.current;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        const width = canvas.width;
        const height = canvas.height;

        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = '#0099ff';

        const barWidth = width / dataArray.length * 2;
        for (let i = 0; i < dataArray.length; i++) {
          const barHeight = (dataArray[i] / 255) * height;
          ctx.fillRect(i * barWidth, height - barHeight, barWidth - 1, barHeight);
        }
      }
    }

    animationFrameRef.current = requestAnimationFrame(updateAudioLevel);
  }, [isRecording]);

  /**
   * WebSocket connection and message handling
   */
  const connectWebSocket = useCallback(async () => {
    try {
      addLog('🔌 Connecting to voice chat server...', 'info', 'client');
      
      const audioInitialized = await initializeAudio();
      if (!audioInitialized) return;

      const ws = new WebSocket(serverUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        startTimeRef.current = new Date().toISOString();
        addLog('✅ Connected to voice chat server', 'success', 'client');
      };

      ws.onmessage = (event) => {
        try {
          const data: WebSocketMessage = JSON.parse(event.data);
          handleWebSocketMessage(data);
        } catch (error: any) {
          addLog(`❌ Message parse error: ${error.message}`, 'error', 'client');
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        setIsRecording(false);
        addLog('🔌 Disconnected from server', 'warning', 'client');
      };

      ws.onerror = (error: any) => {
        addLog(`❌ WebSocket error: ${error.message || 'Connection failed'}`, 'error', 'client');
      };

    } catch (error: any) {
      addLog(`❌ Connection error: ${error.message}`, 'error', 'client');
    }
  }, [serverUrl, initializeAudio, addLog]);

  const handleWebSocketMessage = useCallback((data: WebSocketMessage) => {
    switch (data.type) {
      case 'connected':
        setSessionId(data.session_id);
        setServerInfo(data.server_info);
        addLog(`🎯 Session started: ${data.session_id}`, 'success', 'server');
        if (data.server_info) {
          addLog(`📊 Server config: Real APIs: ${data.server_info.use_real_apis}`, 'info', 'server');
        }
        break;

      case 'transcript':
        addConversationItem(data.transcript, data.response);
        addLog(`📝 Transcript: ${data.transcript}`, 'info', 'server');
        addLog(`🤖 Response: ${data.response}`, 'success', 'server');
        setIsProcessing(false);
        break;

      case 'audio':
        playAudioResponse(data.audio_data);
        break;

      case 'pong':
        addLog('💓 Pong received', 'info', 'server');
        break;

      case 'keepalive':
        // Silent keepalive
        break;

      case 'error':
        addLog(`❌ Server error: ${data.message}`, 'error', 'server');
        setIsProcessing(false);
        break;

      case 'status':
        addLog(`📊 Status: Recording: ${data.is_recording}, Processing: ${data.is_processing}`, 'info', 'server');
        setIsProcessing(data.is_processing);
        break;

      default:
        addLog(`❓ Unknown message type: ${data.type}`, 'warning', 'server');
    }
  }, [addConversationItem, addLog]);

  const playAudioResponse = useCallback(async (audioBase64: string) => {
    try {
      setIsPlaying(true);
      addLog('🔊 Playing AI response...', 'info', 'client');

      const audioData = atob(audioBase64);
      const audioBuffer = new ArrayBuffer(audioData.length);
      const view = new Uint8Array(audioBuffer);

      for (let i = 0; i < audioData.length; i++) {
        view[i] = audioData.charCodeAt(i);
      }

      const blob = new Blob([audioBuffer], { type: 'audio/wav' });
      const audioUrl = URL.createObjectURL(blob);
      const audio = new Audio(audioUrl);

      audio.onended = () => {
        setIsPlaying(false);
        URL.revokeObjectURL(audioUrl);
        addLog('✅ Audio playback completed', 'success', 'client');
      };

      audio.onerror = () => {
        setIsPlaying(false);
        URL.revokeObjectURL(audioUrl);
        addLog('❌ Audio playback failed', 'error', 'client');
      };

      await audio.play();
    } catch (error: any) {
      setIsPlaying(false);
      addLog(`❌ Audio play error: ${error.message}`, 'error', 'client');
    }
  }, [addLog]);

  /**
   * Recording functions
   */
  const startRecording = useCallback(async () => {
    if (!wsRef.current || !streamRef.current) return;

    try {
      setIsRecording(true);
      setIsProcessing(true);
      addLog('🎤 Recording started...', 'info', 'client');

      // Send start recording message
      wsRef.current.send(JSON.stringify({
        type: 'start_recording'
      }));

      // Setup MediaRecorder
      mediaRecorderRef.current = new MediaRecorder(streamRef.current, {
        mimeType: 'audio/webm;codecs=pcm',
        audioBitsPerSecond: 16000
      });

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0 && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          const reader = new FileReader();
          reader.onloadend = () => {
            const base64 = (reader.result as string).split(',')[1];
            wsRef.current!.send(JSON.stringify({
              type: 'audio_data',
              audio_data: base64
            }));
          };
          reader.readAsDataURL(event.data);
        }
      };

      mediaRecorderRef.current.start(1000); // Send data every 100ms
      updateAudioLevel();

    } catch (error: any) {
      setIsRecording(false);
      setIsProcessing(false);
      addLog(`❌ Recording error: ${error.message}`, 'error', 'client');
    }
  }, [addLog, updateAudioLevel]);

  const stopRecording = useCallback(() => {
    if (!mediaRecorderRef.current || !wsRef.current) return;

    try {
      setIsRecording(false);
      addLog('🛑 Recording stopped, processing...', 'info', 'client');

      mediaRecorderRef.current.stop();

      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }

      // Send stop recording message
      wsRef.current.send(JSON.stringify({
        type: 'stop_recording'
      }));

    } catch (error: any) {
      addLog(`❌ Stop recording error: ${error.message}`, 'error', 'client');
      setIsProcessing(false);
    }
  }, [addLog]);

  /**
   * Connection management
   */
  const disconnectConversation = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
    }

    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }

    setIsConnected(false);
    setIsRecording(false);
    setIsPlaying(false);
    setIsProcessing(false);
    setConversationItems([]);
    setSystemLogs([]);
    addLog('👋 Disconnected', 'info', 'client');
  }, [addLog]);

  const sendPing = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'ping' }));
      addLog('💓 Ping sent', 'info', 'client');
    }
  }, [addLog]);

  const getServerStatus = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'get_status' }));
    }
  }, []);

  const deleteConversationItem = useCallback((id: string) => {
    setConversationItems(prev => prev.filter(item => item.id !== id));
  }, []);

  /**
   * Auto-scroll effects
   */
  useEffect(() => {
    if (eventsScrollRef.current) {
      const eventsEl = eventsScrollRef.current;
      const scrollHeight = eventsEl.scrollHeight;
      if (scrollHeight !== eventsScrollHeightRef.current) {
        eventsEl.scrollTop = scrollHeight;
        eventsScrollHeightRef.current = scrollHeight;
      }
    }
  }, [systemLogs]);

  useEffect(() => {
    const conversationEls = [].slice.call(
      document.body.querySelectorAll('[data-conversation-content]')
    );
    for (const el of conversationEls) {
      const conversationEl = el as HTMLDivElement;
      conversationEl.scrollTop = conversationEl.scrollHeight;
    }
  }, [conversationItems]);

  /**
   * Cleanup on unmount
   */
  useEffect(() => {
    return () => {
      disconnectConversation();
    };
  }, [disconnectConversation]);

  /**
   * Server visualization (placeholder for now)
   */
  useEffect(() => {
    let isLoaded = true;

    const render = () => {
      if (isLoaded && serverCanvasRef.current) {
        const canvas = serverCanvasRef.current;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          if (!canvas.width || !canvas.height) {
            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;
          }
          
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          
          if (isPlaying) {
            // Simple visualization for playing state
            const values = new Float32Array(10).fill(Math.random() * 0.8);
            WavRenderer.drawBars(canvas, ctx, values, '#009900', 10, 0, 8);
          }
        }
        window.requestAnimationFrame(render);
      }
    };
    render();

    return () => {
      isLoaded = false;
    };
  }, [isPlaying]);

  /**
   * Render
   */
  return (
    <div data-component="ConsolePage">
      <div className="content-top">
        <div className="content-title">
          <img src="/openai-logomark.svg" />
          <span>voice chat console</span>
        </div>
        <div className="content-api-key">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', color: '#666' }}>
            <span>Server:</span>
            <input
              type="text"
              value={serverUrl}
              onChange={(e) => setServerUrl(e.target.value)}
              disabled={isConnected}
              style={{
                padding: '4px 8px',
                border: '1px solid #ccc',
                borderRadius: '4px',
                fontSize: '12px',
                width: '200px'
              }}
            />
            {sessionId && <span>Session: {sessionId.slice(-8)}</span>}
          </div>
        </div>
      </div>
      
      <div className="content-main">
        <div className="content-logs">
          <div className="content-block events">
            <div className="visualization">
              <div className="visualization-entry client">
                <canvas ref={clientCanvasRef} />
              </div>
              <div className="visualization-entry server">
                <canvas ref={serverCanvasRef} />
              </div>
            </div>
            <div className="content-block-title">system logs</div>
            <div className="content-block-body" ref={eventsScrollRef}>
              {!systemLogs.length && `awaiting connection...`}
              {systemLogs.map((log, i) => (
                <div className="event" key={log.id}>
                  <div className="event-timestamp">
                    {formatTime(log.timestamp)}
                  </div>
                  <div className="event-details">
                    <div className="event-summary">
                      <div className={`event-source ${log.type === 'error' ? 'error' : log.source}`}>
                        {log.source === 'client' ? <ArrowUp /> : 
                         log.source === 'server' ? <ArrowDown /> : 
                         <span>•</span>}
                        <span>{log.source}</span>
                      </div>
                      <div className="event-type" style={{ color: 
                        log.type === 'error' ? '#dc2626' :
                        log.type === 'success' ? '#16a34a' :
                        log.type === 'warning' ? '#ca8a04' : '#6b7280'
                      }}>
                        {log.message}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          <div className="content-block conversation">
            <div className="content-block-title">conversation</div>
            <div className="content-block-body" data-conversation-content>
              {!conversationItems.length && `awaiting connection...`}
              {conversationItems.map((item) => (
                <div className="conversation-item" key={item.id}>
                  <div className="speaker user">
                    <div>you</div>
                    <div className="close" onClick={() => deleteConversationItem(item.id)}>
                      <X />
                    </div>
                  </div>
                  <div className="speaker-content">
                    <div style={{ marginBottom: '8px', padding: '8px', backgroundColor: '#f0f9ff', borderRadius: '4px' }}>
                      {item.transcript}
                    </div>
                    <div style={{ padding: '8px', backgroundColor: '#f0fdf4', borderRadius: '4px' }}>
                      <strong>AI:</strong> {item.response}
                    </div>
                    {item.audio_url && (
                      <audio src={item.audio_url} controls style={{ marginTop: '8px', width: '100%' }} />
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          <div className="content-actions">
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px' }}>
              <span>Mode:</span>
              <Toggle
                defaultValue={useManualMode}
                labels={['auto', 'manual']}
                values={['auto', 'manual']}
                onChange={(enabled) => setUseManualMode(enabled)}
              />
            </div>
            
            <div className="spacer" />
            
            {/* Status indicators */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '12px' }}>
              {isProcessing && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#f59e0b' }}>
                  <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#f59e0b', animation: 'pulse 1s infinite' }}></div>
                  Processing...
                </div>
              )}
              {isPlaying && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#16a34a' }}>
                  <Volume2 size={12} />
                  Playing
                </div>
              )}
              <div>Level: {Math.round(audioLevel * 100)}%</div>
            </div>
            
            <div className="spacer" />
            
            {/* Recording button */}
            {isConnected && useManualMode && (
              <Button
                label={isRecording ? 'release to send' : 'push to talk'}
                buttonStyle={isRecording ? 'alert' : 'regular'}
                disabled={!isConnected || isProcessing}
                onMouseDown={startRecording}
                onMouseUp={stopRecording}
                icon={isRecording ? MicOff : Mic}
              />
            )}
            
            {/* Utility buttons */}
            {isConnected && (
              <>
                <Button
                  label="ping"
                  buttonStyle="flush"
                  onClick={sendPing}
                />
                <Button
                  label="status"
                  buttonStyle="flush"
                  onClick={getServerStatus}
                />
              </>
            )}
            
            <div className="spacer" />
            
            {/* Connect/Disconnect button */}
            <Button
              label={isConnected ? 'disconnect' : 'connect'}
              iconPosition={isConnected ? 'end' : 'start'}
              icon={isConnected ? X : Zap}
              buttonStyle={isConnected ? 'regular' : 'action'}
              onClick={isConnected ? disconnectConversation : connectWebSocket}
            />
          </div>
        </div>
        
        <div className="content-right">
          <div className="content-block kv">
            <div className="content-block-title">server info</div>
            <div className="content-block-body content-kv">
              {serverInfo ? JSON.stringify(serverInfo, null, 2) : 'Not connected'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}