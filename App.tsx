/* ═══════════════════════════════════════════════════════════════════════════
 *  Nexus-Lock™ v2.2.0 — Full FaceNet Integration
 *  © 2026 Nickol Joy Bowman. All Rights Reserved.
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *  CHANGE LOG (v2.2.0 vs v2.1.2)
 *  ─────────────────────────────
 *  • Fixed react-native-mmkv to v4 API (createMMKV instead of new MMKV)
 *  • Fixed face detector API (useFaceDetector + detectFaces, not scanFaces)
 *  • Fixed resize plugin API (useResizePlugin hook, scale: { width, height })
 *  • Fixed babel config (worklets-core/plugin for frame processors)
 *  • Pre-configured FaceNet model download URL (44 MB, 512-dim)
 *  • Added react-native-worklets (Reanimated 4.x peer dep)
 *  • Proper worklets-core bridge for frame processor → JS communication
 *  • Model auto-downloads on first launch, cached locally after that
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator, Alert, FlatList, Pressable,
  SafeAreaView, StatusBar, StyleSheet, Text, TextInput, View,
} from 'react-native';
import * as Haptics from 'expo-haptics';
import * as Crypto from 'expo-crypto';
import * as FileSystem from 'expo-file-system';
import {
  Camera as VisionCamera,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
  runAsync,
  PhotoFile,
} from 'react-native-vision-camera';
import {
  Face,
  useFaceDetector,
  FaceDetectionOptions,
} from 'react-native-vision-camera-face-detector';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import { loadTensorflowModel, TensorflowModel } from 'react-native-fast-tflite';
import * as SQLite from 'expo-sqlite';
import { createMMKV } from 'react-native-mmkv';
import { Worklets } from 'react-native-worklets-core';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';

/* ─────────────────────────────── Types ──────────────────────────────── */

export type FaceList = 'trusted' | 'target';
export type Embedding = number[]; // 512-dim for this FaceNet model

export interface SavedFace {
  id: string;
  name: string;
  list: FaceList;
  photoUri: string;
  embedding: Embedding;   // 512 floats, L2-normalized
  engine: 'facenet';
  createdAt: number;
}

export interface LogEntry {
  id: string;
  kind: 'intrusion' | 'target-detected' | 'auto-unlock' | 'trusted-seen';
  personName?: string;
  photoUri?: string;
  audioUri?: string;
  at: number;
}

export interface AppSettings {
  monitoring: boolean;
  autoUnlock: boolean;
  recordAudio: boolean;
  similarityThreshold: number;  // cosine similarity, 0.65 default
  hasPin: boolean;
  pinHash?: string;
}

interface AppContextType {
  faces: SavedFace[];
  logs: LogEntry[];
  settings: AppSettings;
  isAuthenticated: boolean;
  modelReady: boolean;
  addFace: (face: SavedFace) => void;
  removeFace: (id: string) => void;
  addLog: (entry: LogEntry) => void;
  updateSettings: (patch: Partial<AppSettings>) => void;
  setPin: (pin: string) => Promise<void>;
  verifyPin: (pin: string) => Promise<boolean>;
  authenticate: () => void;
  pruneLogs: (keepDays?: number) => void;
}

/* ─────────────────────────────── Constants ──────────────────────────── */

const EMBEDDING_DIM = 512;

/**
 * FaceNet model — InceptionResnetV1, 512-dim, FP16 quantized (~44 MB).
 * Input:  [1, 160, 160, 3] float32, pixels in [-1, 1]
 * Output: [1, 512] float32, L2-normalized embedding
 *
 * Source: https://github.com/shubham0204/OnDevice-Face-Recognition-Android
 * Backed by the deepface library's FaceNet512 implementation.
 */
const MODEL_DOWNLOAD_URL =
  'https://raw.githubusercontent.com/shubham0204/OnDevice-Face-Recognition-Android/main/app/src/main/assets/facenet_512.tflite';
const MODEL_FILENAME = 'facenet_512.tflite';

const DEFAULT_SETTINGS: AppSettings = {
  monitoring: false,
  autoUnlock: true,
  recordAudio: true,
  similarityThreshold: 0.65,
  hasPin: false,
};

const LOG_RETENTION_DAYS = 30;
const MAX_LOG_ENTRIES = 10_000;
const FRAME_MIN_INTERVAL_MS = 500;       // max 2 fps for monitoring
const ENROLL_CAPTURE_COUNT = 3;           // average 3 embeddings for enrollment
const ENROLL_CAPTURE_INTERVAL_MS = 600;   // wait between enrollment captures

/* ─────────────────────────────── Storage ────────────────────────────── */

/*
 * react-native-mmkv v4 uses createMMKV() (Nitro-based).
 * The old `new MMKV()` constructor from v2 is no longer available.
 */
const mmkv = createMMKV({ id: 'nexuslock-storage' });
const db = SQLite.openDatabaseSync('nexuslock.db');

function initDB(): void {
  db.execSync(`
    CREATE TABLE IF NOT EXISTS logs (
      id TEXT PRIMARY KEY,
      kind TEXT NOT NULL,
      personName TEXT,
      photoUri TEXT,
      audioUri TEXT,
      at INTEGER NOT NULL
    );
  `);
  db.execSync(`CREATE INDEX IF NOT EXISTS idx_logs_at ON logs(at);`);
}

/* ────────────────────── Pure math (exported for tests) ──────────────── */

export function cosine(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na  += a[i] * a[i];
    nb  += b[i] * b[i];
  }
  const d = Math.sqrt(na) * Math.sqrt(nb);
  return d === 0 ? 0 : dot / d;
}

export function bestMatch(
  sig: Embedding,
  faces: SavedFace[],
  threshold: number,
): { face: SavedFace; score: number } | null {
  let best: { face: SavedFace; score: number } | null = null;
  for (const f of faces) {
    if (f.embedding.length !== sig.length) continue;
    const s = cosine(sig, f.embedding);
    if (!best || s > best.score) best = { face: f, score: s };
  }
  return best && best.score >= threshold ? best : null;
}

export function l2Normalize(v: number[]): number[] {
  const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  return n === 0 ? v.slice() : v.map(x => x / n);
}

/** Average multiple embeddings element-wise, then L2-normalize. */
export function averageEmbeddings(embeddings: Embedding[]): Embedding {
  if (embeddings.length === 0) return [];
  if (embeddings.length === 1) return l2Normalize(embeddings[0]);
  const dim = embeddings[0].length;
  const avg = new Array(dim).fill(0);
  for (const e of embeddings) {
    for (let i = 0; i < dim; i++) avg[i] += e[i];
  }
  for (let i = 0; i < dim; i++) avg[i] /= embeddings.length;
  return l2Normalize(avg);
}

/* ────────────────────── PIN hashing (PBKDF2-style) ─────────────────── */

const PIN_SALT = 'nexuslock-pin-salt-v2';

async function hashPin(pin: string): Promise<string> {
  let hash = pin + PIN_SALT;
  for (let i = 0; i < 10_000; i++) {
    hash = await Crypto.digestStringAsync(Crypto.CryptoDigestAlgorithm.SHA256, hash);
  }
  return hash;
}

/* ──────────────────────── FaceNet Engine ────────────────────────────── */

/**
 * Singleton FaceNet engine.
 *
 * Model: InceptionResnetV1 (David Sandberg / deepface)
 * Input:  Float32Array[1 × 160 × 160 × 3]  pixels in [-1, 1]
 * Output: Float32Array[1 × 512]              L2-normalized embedding
 *
 * The `resize` plugin gives us a Uint8Array of RGB [0-255] pixels.
 * We convert to Float32 and normalize: pixel / 127.5 - 1.0
 */
class FaceNetEngine {
  private model: TensorflowModel | null = null;
  private loading = false;
  private _ready = false;

  get ready(): boolean { return this._ready; }

  async load(): Promise<boolean> {
    if (this._ready) return true;
    if (this.loading) return false;
    this.loading = true;
    try {
      const modelDir = FileSystem.documentDirectory + 'models/';
      const modelPath = modelDir + MODEL_FILENAME;

      // Check if model is already cached locally
      const info = await FileSystem.getInfoAsync(modelPath);
      if (!info.exists) {
        console.log('[FaceNet] Downloading model (~44 MB)...');
        await FileSystem.makeDirectoryAsync(modelDir, { intermediates: true });
        const download = await FileSystem.downloadAsync(MODEL_DOWNLOAD_URL, modelPath);
        console.log(`[FaceNet] Download complete: ${download.status}`);
      } else {
        console.log('[FaceNet] Model found in cache');
      }

      // Load from local file
      this.model = await loadTensorflowModel({ url: `file://${modelPath}` });
      this._ready = true;
      console.log('[FaceNet] Model loaded successfully');
      return true;
    } catch (e) {
      console.error('[FaceNet] Failed to load model:', e);
      return false;
    } finally {
      this.loading = false;
    }
  }

  /**
   * Generate a 512-dim embedding from a 160×160 RGB Uint8Array (76,800 bytes).
   * Normalizes pixels from [0,255] → [-1,1] before inference.
   */
  async embed(rgbBuffer: ArrayBuffer): Promise<Embedding | null> {
    if (!this.model) return null;
    try {
      const rgb = new Uint8Array(rgbBuffer);
      const expectedLen = 160 * 160 * 3;
      if (rgb.length < expectedLen) {
        console.warn(`[FaceNet] Buffer too small: ${rgb.length} < ${expectedLen}`);
        return null;
      }

      // Convert Uint8 RGB → Float32 normalized to [-1, 1]
      const input = new Float32Array(expectedLen);
      for (let i = 0; i < expectedLen; i++) {
        input[i] = rgb[i] / 127.5 - 1.0;
      }

      const outputs = await this.model.run([input.buffer]);
      const raw = new Float32Array(outputs[0] as ArrayBuffer);

      if (raw.length !== EMBEDDING_DIM) {
        console.warn(`[FaceNet] Unexpected output dim: ${raw.length}`);
      }

      return l2Normalize(Array.from(raw));
    } catch (e) {
      console.error('[FaceNet] Inference failed:', e);
      return null;
    }
  }
}

const faceNet = new FaceNetEngine();

/* ──────────────────── Face detection options ─────────────────────────── */

const FACE_DETECTION_OPTIONS: FaceDetectionOptions = {
  performanceMode: 'fast',
  classificationMode: 'none',
  landmarkMode: 'none',
  contourMode: 'none',
  minFaceSize: 0.15,
};

/* ──────────────────────── Context + Provider ─────────────────────────── */

const AppContext = React.createContext<AppContextType | null>(null);

function AppProvider({ children }: { children: React.ReactNode }) {
  const [faces, setFaces]       = useState<SavedFace[]>([]);
  const [logs, setLogs]         = useState<LogEntry[]>([]);
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_SETTINGS);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [modelReady, setModelReady] = useState(false);

  // ── Mount: init DB, load persisted state, load model ──
  useEffect(() => {
    initDB();

    const rawFaces = mmkv.getString('faces');
    if (rawFaces) { try { setFaces(JSON.parse(rawFaces)); } catch {} }

    const rawSettings = mmkv.getString('settings');
    if (rawSettings) {
      try { setSettings(prev => ({ ...prev, ...JSON.parse(rawSettings) })); } catch {}
    }

    try {
      const rows = db.getAllSync<LogEntry>('SELECT * FROM logs ORDER BY at DESC LIMIT 500');
      setLogs(rows);
    } catch (e) { console.error('Failed to load logs:', e); }

    // Eagerly load FaceNet
    faceNet.load().then(ok => setModelReady(ok));
  }, []);

  // Auto-prune
  useEffect(() => { pruneLogs(LOG_RETENTION_DAYS); }, []);

  const addFace = useCallback((face: SavedFace) => {
    setFaces(prev => {
      const next = [...prev, face];
      mmkv.set('faces', JSON.stringify(next));
      return next;
    });
  }, []);

  const removeFace = useCallback((id: string) => {
    setFaces(prev => {
      const next = prev.filter(f => f.id !== id);
      mmkv.set('faces', JSON.stringify(next));
      return next;
    });
  }, []);

  const addLog = useCallback((entry: LogEntry) => {
    setLogs(prev => [entry, ...prev].slice(0, MAX_LOG_ENTRIES));
    try {
      db.runSync(
        'INSERT OR REPLACE INTO logs (id,kind,personName,photoUri,audioUri,at) VALUES (?,?,?,?,?,?)',
        [entry.id, entry.kind, entry.personName ?? null, entry.photoUri ?? null, entry.audioUri ?? null, entry.at],
      );
    } catch (e) { console.error('Failed to persist log:', e); }
  }, []);

  const updateSettings = useCallback((patch: Partial<AppSettings>) => {
    setSettings(prev => {
      const next = { ...prev, ...patch };
      mmkv.set('settings', JSON.stringify(next));
      return next;
    });
  }, []);

  const setPin = useCallback(async (pin: string) => {
    const pinHash = await hashPin(pin);
    updateSettings({ hasPin: true, pinHash });
  }, [updateSettings]);

  const verifyPin = useCallback(async (pin: string): Promise<boolean> => {
    const hash = await hashPin(pin);
    return hash === settings.pinHash;
  }, [settings.pinHash]);

  const authenticate = useCallback(() => setIsAuthenticated(true), []);

  const pruneLogs = useCallback((keepDays: number = LOG_RETENTION_DAYS) => {
    const cutoff = Date.now() - keepDays * 86_400_000;
    try {
      db.runSync('DELETE FROM logs WHERE at < ?', [cutoff]);
      setLogs(prev => prev.filter(l => l.at >= cutoff));
    } catch (e) { console.error('Prune failed:', e); }
  }, []);

  const ctx = useMemo<AppContextType>(() => ({
    faces, logs, settings, isAuthenticated, modelReady,
    addFace, removeFace, addLog, updateSettings,
    setPin, verifyPin, authenticate, pruneLogs,
  }), [faces, logs, settings, isAuthenticated, modelReady,
       addFace, removeFace, addLog, updateSettings,
       setPin, verifyPin, authenticate, pruneLogs]);

  return <AppContext.Provider value={ctx}>{children}</AppContext.Provider>;
}

function useApp(): AppContextType {
  const c = React.useContext(AppContext);
  if (!c) throw new Error('useApp must be used within AppProvider');
  return c;
}

/* ──────────────────────── Monitor Canvas ─────────────────────────────── */

function MonitorCanvas() {
  const device = useCameraDevice('front');
  const { hasPermission, requestPermission } = useCameraPermission();
  const { faces, settings, addLog } = useApp();

  const busyRef    = useRef(false);
  const lastTsRef  = useRef(0);

  // Use refs so JS callback always has latest state
  const facesRef    = useRef(faces);
  const settingsRef = useRef(settings);
  useEffect(() => { facesRef.current = faces; }, [faces]);
  useEffect(() => { settingsRef.current = settings; }, [settings]);

  /*
   * Face detector — useFaceDetector() hook from
   * react-native-vision-camera-face-detector.
   * Returns detectFaces (worklet callable) and stopListeners (cleanup).
   */
  const faceDetectionOptions = useRef<FaceDetectionOptions>(FACE_DETECTION_OPTIONS).current;
  const { detectFaces, stopListeners } = useFaceDetector(faceDetectionOptions);

  useEffect(() => {
    return () => { stopListeners(); };
  }, [stopListeners]);

  /*
   * Resize plugin — useResizePlugin() hook from vision-camera-resize-plugin.
   * Returns a `resize` function callable inside frame processor worklets.
   */
  const { resize } = useResizePlugin();

  /*
   * Bridge: worklet → JS thread.
   * Worklets.createRunOnJS (from react-native-worklets-core) creates a
   * function that can be called from within a VisionCamera frame processor
   * worklet and runs the callback on the JS thread.
   */
  const handleDetection = Worklets.createRunOnJS(async (rgbBuffer: ArrayBuffer) => {
    if (busyRef.current) return;
    const now = Date.now();
    if (now - lastTsRef.current < FRAME_MIN_INTERVAL_MS) return;
    busyRef.current = true;
    lastTsRef.current = now;

    try {
      const embedding = await faceNet.embed(rgbBuffer);
      if (!embedding || embedding.length !== EMBEDDING_DIM) return;

      const currentFaces    = facesRef.current;
      const currentSettings = settingsRef.current;
      const match = bestMatch(embedding, currentFaces, currentSettings.similarityThreshold);

      if (match) {
        if (match.face.list === 'trusted') {
          addLog({ id: Crypto.randomUUID(), kind: 'trusted-seen', personName: match.face.name, at: Date.now() });
        } else {
          addLog({ id: Crypto.randomUUID(), kind: 'target-detected', personName: match.face.name, at: Date.now() });
          Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
          Alert.alert('⚠️ Target Detected', `${match.face.name} was detected.`);
        }
      } else if (currentFaces.length > 0) {
        addLog({ id: Crypto.randomUUID(), kind: 'intrusion', at: Date.now() });
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
      }
    } catch (e) {
      console.error('[Monitor] Frame error:', e);
    } finally {
      busyRef.current = false;
    }
  });

  /**
   * Frame processor pipeline:
   *   1. detectFaces(frame) — ML Kit face detection (native, fast)
   *   2. If face found → resize to 160×160 RGB uint8 (native, fast)
   *   3. Send resized buffer to JS thread via Worklets bridge
   *   4. JS thread: normalize → FaceNet inference → match → log
   *
   * Using runAsync so face detection + resize don't block the camera pipeline.
   */
  const frameProcessor = useFrameProcessor((frame) => {
    'worklet';
    runAsync(frame, () => {
      'worklet';
      try {
        const detected = detectFaces(frame);
        if (detected.length === 0) return;

        // Resize entire frame to 160×160 RGB (center-crop + scale)
        const resized = resize(frame, {
          scale: { width: 160, height: 160 },
          pixelFormat: 'rgb',
          dataType: 'uint8',
        });

        // Send to JS thread for inference
        handleDetection(resized.buffer);
      } catch {
        // Silently drop frames that fail
      }
    });
  }, [detectFaces, resize, handleDetection]);

  if (!hasPermission) {
    return (
      <View style={styles.permissionBox}>
        <Text style={styles.permText}>Camera access is required for face detection</Text>
        <Pressable style={styles.btn} onPress={requestPermission}>
          <Text style={styles.btnText}>Grant Camera Permission</Text>
        </Pressable>
      </View>
    );
  }
  if (!device) {
    return <View style={styles.permissionBox}><Text style={styles.permText}>No front camera found</Text></View>;
  }

  return (
    <VisionCamera
      device={device}
      frameProcessor={frameProcessor}
      style={StyleSheet.absoluteFill}
      isActive={settings.monitoring}
      pixelFormat="yuv"
    />
  );
}

/* ──────────────────────── Error Boundary ─────────────────────────────── */

class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error?: Error }
> {
  state = { hasError: false, error: undefined as Error | undefined };
  static getDerivedStateFromError(error: Error) { return { hasError: true, error }; }

  render() {
    if (this.state.hasError) {
      return (
        <View style={[styles.screen, { justifyContent: 'center', alignItems: 'center', padding: 20 }]}>
          <Text style={[styles.h1, { textAlign: 'center' }]}>Something went wrong</Text>
          <Text style={[styles.sub, { textAlign: 'center', marginTop: 12 }]}>
            {this.state.error?.message ?? 'Unknown error'}
          </Text>
          <Pressable style={[styles.btn, { marginTop: 20 }]}
            onPress={() => this.setState({ hasError: false, error: undefined })}>
            <Text style={styles.btnText}>Try Again</Text>
          </Pressable>
        </View>
      );
    }
    return this.props.children;
  }
}

/* ──────────────────────── PIN Screen ────────────────────────────────── */

function PinScreen({ onSuccess }: { onSuccess: () => void }) {
  const { verifyPin } = useApp();
  const [pin, setPin] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  const submit = useCallback(async () => {
    if (pin.length < 4) { setError('PIN must be at least 4 digits'); return; }
    setBusy(true); setError('');
    try {
      if (await verifyPin(pin)) { onSuccess(); }
      else { setError('Incorrect PIN'); setPin(''); Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error); }
    } catch { setError('Verification failed'); }
    finally { setBusy(false); }
  }, [pin, verifyPin, onSuccess]);

  return (
    <SafeAreaView style={styles.screen}>
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', padding: 40 }}>
        <Text style={styles.h1}>🔒 Nexus-Lock</Text>
        <Text style={[styles.sub, { marginBottom: 30 }]}>Enter your PIN</Text>
        <TextInput style={styles.pinInput} value={pin} onChangeText={setPin}
          keyboardType="number-pad" secureTextEntry maxLength={8}
          placeholder="• • • •" placeholderTextColor="#7A8699" autoFocus onSubmitEditing={submit} />
        {error ? <Text style={styles.errorText}>{error}</Text> : null}
        <Pressable style={[styles.btn, { marginTop: 20, opacity: busy ? 0.5 : 1 }]}
          onPress={submit} disabled={busy}>
          {busy ? <ActivityIndicator color="#0A0E14" /> : <Text style={styles.btnText}>Unlock</Text>}
        </Pressable>
      </View>
    </SafeAreaView>
  );
}

/* ──────────────────────── Settings Screen ───────────────────────────── */

function SettingsScreen() {
  const { settings, updateSettings, setPin, faces, removeFace } = useApp();
  const [newPin, setNewPin] = useState('');

  const handleSetPin = useCallback(async () => {
    if (newPin.length < 4) { Alert.alert('Error', 'PIN must be at least 4 digits'); return; }
    await setPin(newPin); setNewPin('');
    Alert.alert('Success', 'PIN has been set');
  }, [newPin, setPin]);

  return (
    <SafeAreaView style={styles.screen}>
      <View style={{ padding: 20, flex: 1 }}>
        <Text style={styles.h1}>Settings</Text>

        <SettingRow label="Monitoring" value={settings.monitoring}
          onToggle={() => updateSettings({ monitoring: !settings.monitoring })} />
        <SettingRow label="Auto-Unlock (trusted faces)" value={settings.autoUnlock}
          onToggle={() => updateSettings({ autoUnlock: !settings.autoUnlock })} />
        <SettingRow label="Record Audio on Intrusion" value={settings.recordAudio}
          onToggle={() => updateSettings({ recordAudio: !settings.recordAudio })} />

        <Text style={[styles.sub, { marginTop: 24, marginBottom: 4 }]}>
          Similarity Threshold: {settings.similarityThreshold.toFixed(2)}
        </Text>
        <View style={{ flexDirection: 'row', gap: 10, marginTop: 8 }}>
          <Pressable style={[styles.btn, { flex: 1, backgroundColor: '#1A1F2B' }]}
            onPress={() => updateSettings({ similarityThreshold: Math.max(0.3, settings.similarityThreshold - 0.05) })}>
            <Text style={[styles.btnText, { color: '#E6F1FF' }]}>– Lower</Text>
          </Pressable>
          <Pressable style={[styles.btn, { flex: 1, backgroundColor: '#1A1F2B' }]}
            onPress={() => updateSettings({ similarityThreshold: Math.min(0.95, settings.similarityThreshold + 0.05) })}>
            <Text style={[styles.btnText, { color: '#E6F1FF' }]}>+ Raise</Text>
          </Pressable>
        </View>

        <Text style={[styles.h2, { marginTop: 28 }]}>{settings.hasPin ? 'Change PIN' : 'Set PIN'}</Text>
        <TextInput style={styles.pinInput} value={newPin} onChangeText={setNewPin}
          keyboardType="number-pad" secureTextEntry maxLength={8}
          placeholder="Enter 4-8 digit PIN" placeholderTextColor="#7A8699" />
        <Pressable style={[styles.btn, { marginTop: 12 }]} onPress={handleSetPin}>
          <Text style={styles.btnText}>{settings.hasPin ? 'Update PIN' : 'Set PIN'}</Text>
        </Pressable>

        {/* Enrolled faces list with delete */}
        {faces.length > 0 && (
          <>
            <Text style={[styles.h2, { marginTop: 28 }]}>Enrolled Faces ({faces.length})</Text>
            {faces.map(f => (
              <View key={f.id} style={{ flexDirection: 'row', justifyContent: 'space-between',
                alignItems: 'center', paddingVertical: 10, borderBottomWidth: 1, borderBottomColor: '#1A1F2B' }}>
                <View>
                  <Text style={{ color: '#E6F1FF', fontSize: 16 }}>{f.name}</Text>
                  <Text style={{ color: '#7A8699', fontSize: 12 }}>
                    {f.list} · {f.embedding.length}-dim · {new Date(f.createdAt).toLocaleDateString()}
                  </Text>
                </View>
                <Pressable onPress={() => {
                  Alert.alert('Remove Face', `Delete "${f.name}"?`, [
                    { text: 'Cancel', style: 'cancel' },
                    { text: 'Delete', style: 'destructive', onPress: () => removeFace(f.id) },
                  ]);
                }}>
                  <Text style={{ color: '#FF4444', fontSize: 14 }}>Delete</Text>
                </Pressable>
              </View>
            ))}
          </>
        )}
      </View>
    </SafeAreaView>
  );
}

function SettingRow({ label, value, onToggle }: { label: string; value: boolean; onToggle: () => void }) {
  return (
    <Pressable style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
      paddingVertical: 14, borderBottomWidth: 1, borderBottomColor: '#1A1F2B' }} onPress={onToggle}>
      <Text style={{ color: '#E6F1FF', fontSize: 16 }}>{label}</Text>
      <Text style={{ color: value ? '#00FFAA' : '#7A8699', fontSize: 16, fontWeight: '600' }}>
        {value ? 'ON' : 'OFF'}
      </Text>
    </Pressable>
  );
}

/* ──────────────────── Face Enrollment Screen ────────────────────────── */

/**
 * Enrollment uses the *frame processor* (not takePhoto) to generate
 * real FaceNet embeddings. Flow:
 *
 *   1. User enters name, picks trusted/target
 *   2. Taps "Start Enrollment" → camera activates frame processor
 *   3. Frame processor detects a face, resizes to 160×160, sends to JS
 *   4. JS runs FaceNet inference → gets 512-dim embedding
 *   5. Repeats 3 times, averages the embeddings for robustness
 *   6. Also captures a photo for display
 *   7. Saves face with averaged embedding
 */
function EnrollScreen({ navigation }: any) {
  const device = useCameraDevice('front');
  const { hasPermission } = useCameraPermission();
  const { addFace, modelReady } = useApp();
  const cameraRef = useRef<typeof VisionCamera>(null);

  const [name, setName]       = useState('');
  const [list, setList]       = useState<FaceList>('trusted');
  const [phase, setPhase]     = useState<'idle' | 'capturing' | 'processing' | 'done'>('idle');
  const [progress, setProgress] = useState(0);

  const capturedEmbeddings = useRef<Embedding[]>([]);
  const enrollingRef       = useRef(false);

  /*
   * Face detector + resize hooks (same as MonitorCanvas)
   */
  const faceDetectionOptions = useRef<FaceDetectionOptions>(FACE_DETECTION_OPTIONS).current;
  const { detectFaces, stopListeners } = useFaceDetector(faceDetectionOptions);
  const { resize } = useResizePlugin();

  useEffect(() => {
    return () => { stopListeners(); };
  }, [stopListeners]);

  const handleEnrollFrame = Worklets.createRunOnJS(async (rgbBuffer: ArrayBuffer) => {
    if (!enrollingRef.current) return;
    if (capturedEmbeddings.current.length >= ENROLL_CAPTURE_COUNT) return;

    const embedding = await faceNet.embed(rgbBuffer);
    if (!embedding || embedding.length !== EMBEDDING_DIM) return;

    capturedEmbeddings.current.push(embedding);
    const count = capturedEmbeddings.current.length;
    setProgress(count);

    if (count >= ENROLL_CAPTURE_COUNT) {
      enrollingRef.current = false;
      setPhase('processing');

      // Average the captured embeddings
      const averaged = averageEmbeddings(capturedEmbeddings.current);

      // Also take a photo for display
      let photoUri = '';
      try {
        if (cameraRef.current) {
          const photo: PhotoFile = await (cameraRef.current as any).takePhoto({ flash: 'off' });
          photoUri = `file://${photo.path}`;
        }
      } catch (e) {
        console.warn('[Enroll] Photo capture failed, proceeding without:', e);
      }

      addFace({
        id: Crypto.randomUUID(),
        name: name.trim(),
        list,
        photoUri,
        embedding: averaged,
        engine: 'facenet',
        createdAt: Date.now(),
      });

      setPhase('done');
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      Alert.alert(
        '✅ Enrolled!',
        `"${name.trim()}" saved as ${list} with ${EMBEDDING_DIM}-dim embedding (averaged from ${ENROLL_CAPTURE_COUNT} captures).`,
        [{ text: 'OK', onPress: () => navigation.goBack() }],
      );
    }
  });

  const frameProcessor = useFrameProcessor((frame) => {
    'worklet';
    runAsync(frame, () => {
      'worklet';
      try {
        const detected = detectFaces(frame);
        if (detected.length === 0) return;

        const resized = resize(frame, {
          scale: { width: 160, height: 160 },
          pixelFormat: 'rgb',
          dataType: 'uint8',
        });

        handleEnrollFrame(resized.buffer);
      } catch {}
    });
  }, [detectFaces, resize, handleEnrollFrame]);

  const startEnrollment = useCallback(() => {
    if (!name.trim()) { Alert.alert('Error', 'Enter a name'); return; }
    if (!modelReady)  { Alert.alert('Error', 'FaceNet model is still loading. Please wait.'); return; }
    capturedEmbeddings.current = [];
    enrollingRef.current = true;
    setProgress(0);
    setPhase('capturing');
  }, [name, modelReady]);

  if (!hasPermission || !device) {
    return (
      <SafeAreaView style={styles.screen}>
        <View style={{ padding: 20 }}>
          <Text style={styles.h1}>Enroll Face</Text>
          <Text style={styles.sub}>Camera permission required</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.screen}>
      <View style={{ flex: 1 }}>
        {/* Camera preview */}
        <View style={{ flex: 1, position: 'relative' }}>
          <VisionCamera
            ref={cameraRef as any}
            device={device}
            photo={true}
            frameProcessor={phase === 'capturing' ? frameProcessor : undefined}
            style={StyleSheet.absoluteFill}
            isActive={phase === 'idle' || phase === 'capturing'}
            pixelFormat="yuv"
          />

          {/* Overlay */}
          <View style={{ position: 'absolute', top: 16, left: 16, right: 16 }}>
            <Text style={styles.h1}>Enroll Face</Text>
            {!modelReady && (
              <View style={{ flexDirection: 'row', alignItems: 'center', marginTop: 8 }}>
                <ActivityIndicator size="small" color="#00FFAA" />
                <Text style={[styles.sub, { marginLeft: 8, color: '#FFD700' }]}>Loading FaceNet model...</Text>
              </View>
            )}
          </View>

          {phase === 'capturing' && (
            <View style={{ position: 'absolute', bottom: 16, left: 16, right: 16, alignItems: 'center' }}>
              <Text style={{ color: '#00FFAA', fontSize: 20, fontWeight: '700' }}>
                Capturing... {progress}/{ENROLL_CAPTURE_COUNT}
              </Text>
              <Text style={{ color: '#7A8699', fontSize: 14, marginTop: 4 }}>
                Look at the camera · keep your face centered
              </Text>
              {/* Progress bar */}
              <View style={{ width: '100%', height: 6, backgroundColor: '#1A1F2B', borderRadius: 3, marginTop: 12 }}>
                <View style={{ width: `${(progress / ENROLL_CAPTURE_COUNT) * 100}%`, height: 6,
                  backgroundColor: '#00FFAA', borderRadius: 3 }} />
              </View>
            </View>
          )}

          {phase === 'processing' && (
            <View style={[StyleSheet.absoluteFill, { justifyContent: 'center', alignItems: 'center', backgroundColor: 'rgba(10,14,20,0.8)' }]}>
              <ActivityIndicator size="large" color="#00FFAA" />
              <Text style={{ color: '#E6F1FF', fontSize: 18, marginTop: 12 }}>Processing embeddings...</Text>
            </View>
          )}
        </View>

        {/* Controls */}
        <View style={{ padding: 20, backgroundColor: '#0A0E14' }}>
          <TextInput style={[styles.pinInput, { letterSpacing: 0, fontSize: 18 }]}
            value={name} onChangeText={setName}
            placeholder="Name (e.g. John)" placeholderTextColor="#7A8699"
            editable={phase === 'idle'} />

          <View style={{ flexDirection: 'row', gap: 10, marginTop: 10 }}>
            <Pressable
              style={[styles.btn, { flex: 1, backgroundColor: list === 'trusted' ? '#00FFAA' : '#1A1F2B' }]}
              onPress={() => phase === 'idle' && setList('trusted')}>
              <Text style={[styles.btnText, { color: list === 'trusted' ? '#0A0E14' : '#7A8699' }]}>Trusted</Text>
            </Pressable>
            <Pressable
              style={[styles.btn, { flex: 1, backgroundColor: list === 'target' ? '#FF4444' : '#1A1F2B' }]}
              onPress={() => phase === 'idle' && setList('target')}>
              <Text style={[styles.btnText, { color: list === 'target' ? '#FFF' : '#7A8699' }]}>Target</Text>
            </Pressable>
          </View>

          {phase === 'idle' && (
            <Pressable
              style={[styles.btn, { marginTop: 10, opacity: modelReady ? 1 : 0.5 }]}
              onPress={startEnrollment} disabled={!modelReady}>
              <Text style={styles.btnText}>
                {modelReady ? '📸 Start Enrollment' : '⏳ Loading Model...'}
              </Text>
            </Pressable>
          )}
        </View>
      </View>
    </SafeAreaView>
  );
}

/* ──────────────────────── Logs Screen ──────────────────────────────── */

function LogsScreen() {
  const { logs, pruneLogs } = useApp();
  const icons: Record<string, string> = {
    'intrusion': '🚨', 'target-detected': '⚠️', 'auto-unlock': '🔓', 'trusted-seen': '✅',
  };

  const renderLog = useCallback(({ item }: { item: LogEntry }) => (
    <View style={{ paddingVertical: 12, borderBottomWidth: 1, borderBottomColor: '#1A1F2B' }}>
      <Text style={{ color: '#E6F1FF', fontSize: 15 }}>
        {icons[item.kind] ?? '📋'} {item.kind}{item.personName ? ` — ${item.personName}` : ''}
      </Text>
      <Text style={{ color: '#7A8699', fontSize: 12, marginTop: 2 }}>{new Date(item.at).toLocaleString()}</Text>
    </View>
  ), []);

  return (
    <SafeAreaView style={styles.screen}>
      <View style={{ flex: 1, padding: 20 }}>
        <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
          <Text style={styles.h1}>Event Log</Text>
          <Pressable onPress={() => Alert.alert('Prune Logs', 'Delete logs older than 30 days?', [
            { text: 'Cancel', style: 'cancel' },
            { text: 'Prune', style: 'destructive', onPress: () => pruneLogs(30) },
          ])}>
            <Text style={{ color: '#FF4444', fontSize: 14 }}>Prune</Text>
          </Pressable>
        </View>
        {logs.length === 0
          ? <Text style={[styles.sub, { marginTop: 20 }]}>No events recorded yet</Text>
          : <FlatList data={logs} keyExtractor={i => i.id} renderItem={renderLog}
              initialNumToRender={20} maxToRenderPerBatch={20} />}
      </View>
    </SafeAreaView>
  );
}

/* ──────────────────────── Home Screen ──────────────────────────────── */

function HomeScreen({ navigation }: any) {
  const { settings, updateSettings, faces, logs, modelReady } = useApp();

  return (
    <SafeAreaView style={styles.screen}>
      <View style={{ flex: 1, position: 'relative' }}>
        {settings.monitoring ? (
          <MonitorCanvas />
        ) : (
          <View style={[StyleSheet.absoluteFill, { justifyContent: 'center', alignItems: 'center', backgroundColor: '#0D1117' }]}>
            <Text style={{ color: '#7A8699', fontSize: 48 }}>📷</Text>
            <Text style={[styles.sub, { marginTop: 8 }]}>Monitoring paused</Text>
          </View>
        )}
        <View style={{ position: 'absolute', top: 16, left: 16, right: 16 }}>
          <Text style={styles.h1}>Nexus-Lock</Text>
          <Text style={styles.sub}>
            {faces.length} face{faces.length !== 1 ? 's' : ''} enrolled · {logs.length} event{logs.length !== 1 ? 's' : ''}
          </Text>
          {!modelReady && (
            <View style={{ flexDirection: 'row', alignItems: 'center', marginTop: 6 }}>
              <ActivityIndicator size="small" color="#FFD700" />
              <Text style={{ color: '#FFD700', fontSize: 13, marginLeft: 6 }}>Loading FaceNet model...</Text>
            </View>
          )}
          {modelReady && (
            <Text style={{ color: '#00FFAA', fontSize: 13, marginTop: 6 }}>✓ FaceNet ready ({EMBEDDING_DIM}-dim)</Text>
          )}
        </View>
      </View>

      <View style={{ padding: 16, backgroundColor: '#0A0E14' }}>
        <Pressable
          style={[styles.btn, { backgroundColor: settings.monitoring ? '#FF4444' : '#00FFAA', marginBottom: 10 }]}
          onPress={() => updateSettings({ monitoring: !settings.monitoring })}>
          <Text style={[styles.btnText, { color: settings.monitoring ? '#FFF' : '#0A0E14' }]}>
            {settings.monitoring ? '⏹ Stop Monitoring' : '▶ Start Monitoring'}
          </Text>
        </Pressable>
        <View style={{ flexDirection: 'row', gap: 10 }}>
          <Pressable style={[styles.btn, { flex: 1, backgroundColor: '#1A1F2B' }]}
            onPress={() => navigation.navigate('Enroll')}>
            <Text style={[styles.btnText, { color: '#E6F1FF' }]}>+ Enroll</Text>
          </Pressable>
          <Pressable style={[styles.btn, { flex: 1, backgroundColor: '#1A1F2B' }]}
            onPress={() => navigation.navigate('Logs')}>
            <Text style={[styles.btnText, { color: '#E6F1FF' }]}>📋 Logs</Text>
          </Pressable>
          <Pressable style={[styles.btn, { flex: 1, backgroundColor: '#1A1F2B' }]}
            onPress={() => navigation.navigate('Settings')}>
            <Text style={[styles.btnText, { color: '#E6F1FF' }]}>⚙️</Text>
          </Pressable>
        </View>
      </View>
    </SafeAreaView>
  );
}

/* ──────────────────────── Navigation ────────────────────────────────── */

const Stack = createNativeStackNavigator();

function AuthenticatedApp() {
  return (
    <NavigationContainer>
      <Stack.Navigator screenOptions={{ headerShown: false, contentStyle: { backgroundColor: '#0A0E14' } }}>
        <Stack.Screen name="Home" component={HomeScreen} />
        <Stack.Screen name="Enroll" component={EnrollScreen} />
        <Stack.Screen name="Logs" component={LogsScreen} />
        <Stack.Screen name="Settings" component={SettingsScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

function AppInner() {
  const { settings, isAuthenticated, authenticate } = useApp();
  if (settings.hasPin && !isAuthenticated) return <PinScreen onSuccess={authenticate} />;
  return <AuthenticatedApp />;
}

export default function App() {
  return (
    <View style={{ flex: 1, backgroundColor: '#0A0E14' }}>
      <StatusBar barStyle="light-content" />
      <ErrorBoundary>
        <AppProvider>
          <AppInner />
        </AppProvider>
      </ErrorBoundary>
    </View>
  );
}

/* ──────────────────────── Styles ────────────────────────────────────── */

const styles = StyleSheet.create({
  screen:        { flex: 1, backgroundColor: '#0A0E14' },
  h1:            { color: '#E6F1FF', fontSize: 28, fontWeight: '700' },
  h2:            { color: '#E6F1FF', fontSize: 20, fontWeight: '600' },
  sub:           { color: '#7A8699', fontSize: 14, marginTop: 4 },
  btn:           { backgroundColor: '#00FFAA', paddingVertical: 14, paddingHorizontal: 20, borderRadius: 10, alignItems: 'center' },
  btnText:       { color: '#0A0E14', fontSize: 16, fontWeight: '600' },
  pinInput:      { backgroundColor: '#1A1F2B', color: '#E6F1FF', fontSize: 24, textAlign: 'center',
                   paddingVertical: 14, paddingHorizontal: 20, borderRadius: 10, letterSpacing: 8, marginTop: 10 },
  permissionBox: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 40, backgroundColor: '#0D1117' },
  permText:      { color: '#7A8699', fontSize: 16, textAlign: 'center', marginBottom: 16 },
  errorText:     { color: '#FF4444', fontSize: 14, marginTop: 8 },
});
