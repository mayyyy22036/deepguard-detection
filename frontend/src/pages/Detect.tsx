import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ScanFace, Image } from "lucide-react";
import axios from "axios";
import FileUpload from "@/components/FileUpload";
import ResultDisplay from "@/components/ResultDisplay";
import LoadingSpinner from "@/components/LoadingSpinner";

const API_BASE = "http://localhost:8000";

type ApiResponse = {
  prediction: "Real" | "Fake";
  confidence: number;
  processing_time: number;
  probabilities: { real: number; fake: number };
};

const Detect = () => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<ApiResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelect = useCallback((f: File) => {
    setFile(f);
    setResult(null);
    setError(null);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    if (f.type.startsWith("video/")) {
      setPreview(URL.createObjectURL(f));
    } else {
      reader.readAsDataURL(f);
    }
  }, []);

  const analyze = useCallback(async () => {
    if (!file) return;
    setAnalyzing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const { data } = await axios.post<ApiResponse>(`${API_BASE}/predict`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 30000,
      });

      setResult(data);
    } catch (err) {
      if (axios.isAxiosError(err)) {
        if (err.code === "ERR_NETWORK" || err.code === "ECONNREFUSED") {
          setError("Cannot connect to the API server. Please make sure the backend is running at " + API_BASE);
        } else if (err.response) {
          setError(`Server error: ${err.response.status} — ${err.response.data?.detail || "Unknown error"}`);
        } else {
          setError("Request timed out. Please try again.");
        }
      } else {
        setError("An unexpected error occurred. Please try again.");
      }
    } finally {
      setAnalyzing(false);
    }
  }, [file]);

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="min-h-screen py-12">
      <div className="container mx-auto px-4 max-w-5xl">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="text-center mb-10">
          <h1 className="text-4xl font-bold mb-2">Deepfake Detection</h1>
          <p className="text-muted-foreground">
            Upload an image or video to analyze whether it's authentic or manipulated.
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Upload */}
          <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 }}>
            <FileUpload
              file={file}
              preview={preview}
              onFileSelect={handleFileSelect}
              onClear={reset}
              disabled={analyzing}
            />

            {file && !analyzing && !result && (
              <button
                onClick={analyze}
                className="mt-4 w-full bg-gradient-primary text-primary-foreground font-semibold py-3 rounded-xl shadow-glow hover:shadow-glow-lg transition-all flex items-center justify-center gap-2"
              >
                <ScanFace className="h-5 w-5" />
                Analyze
              </button>
            )}

            {error && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="mt-4 p-4 rounded-xl border border-destructive/30 bg-destructive/5 text-destructive text-sm"
              >
                {error}
              </motion.div>
            )}
          </motion.div>

          {/* Results */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
          >
            <AnimatePresence mode="wait">
              {!result && !analyzing && (
                <motion.div
                  key="placeholder"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="flex flex-col items-center justify-center h-full min-h-[300px] border border-border rounded-xl bg-card p-8"
                >
                  <Image className="h-16 w-16 text-muted-foreground/30 mb-4" />
                  <p className="text-muted-foreground text-center text-sm">
                    Analysis results will appear here
                  </p>
                </motion.div>
              )}

              {analyzing && (
                <motion.div
                  key="loading"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="flex items-center justify-center h-full min-h-[300px] border border-border rounded-xl bg-card"
                >
                  <LoadingSpinner text="Analyzing..." />
                </motion.div>
              )}

              {result && (
                <ResultDisplay
                  prediction={result.prediction}
                  confidence={result.confidence}
                  processingTime={result.processing_time}
                  probabilities={result.probabilities}
                  onReset={reset}
                />
              )}
            </AnimatePresence>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default Detect;
