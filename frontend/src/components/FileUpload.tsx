import { useCallback, useState } from "react";
import { Upload, X, FileImage, Film } from "lucide-react";

interface FileUploadProps {
  file: File | null;
  preview: string | null;
  onFileSelect: (file: File) => void;
  onClear: () => void;
  disabled?: boolean;
}

const MAX_SIZE = 10 * 1024 * 1024; // 10MB
const ACCEPTED_TYPES = ["image/jpeg", "image/png", "video/mp4"];

const FileUpload = ({ file, preview, onFileSelect, onClear, disabled }: FileUploadProps) => {
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const validateAndSet = useCallback(
    (f: File) => {
      setError(null);
      if (!ACCEPTED_TYPES.includes(f.type)) {
        setError("Only JPG, PNG images and MP4 videos are accepted.");
        return;
      }
      if (f.size > MAX_SIZE) {
        setError("File size must be under 10MB.");
        return;
      }
      onFileSelect(f);
    },
    [onFileSelect]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      if (e.dataTransfer.files[0]) validateAndSet(e.dataTransfer.files[0]);
    },
    [validateAndSet]
  );

  if (file && preview) {
    const isVideo = file.type.startsWith("video/");
    return (
      <div className="relative rounded-xl overflow-hidden border border-border bg-card">
        {isVideo ? (
          <video src={preview} className="w-full h-72 object-cover" controls />
        ) : (
          <img src={preview} alt="Preview" className="w-full h-72 object-cover" />
        )}
        {!disabled && (
          <button
            onClick={onClear}
            className="absolute top-3 right-3 p-1.5 rounded-full bg-background/80 text-foreground hover:bg-destructive hover:text-destructive-foreground transition-colors"
          >
            <X className="h-4 w-4" />
          </button>
        )}
        <div className="absolute bottom-0 left-0 right-0 bg-background/80 backdrop-blur-sm px-4 py-2 flex items-center gap-2">
          {isVideo ? <Film className="h-4 w-4 text-primary" /> : <FileImage className="h-4 w-4 text-primary" />}
          <span className="text-sm text-foreground truncate">{file.name}</span>
          <span className="text-xs text-muted-foreground ml-auto">{(file.size / 1024 / 1024).toFixed(1)} MB</span>
        </div>
      </div>
    );
  }

  return (
    <div>
      <label
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        className={`flex flex-col items-center justify-center h-72 border-2 border-dashed rounded-xl cursor-pointer transition-all ${
          dragOver
            ? "border-primary bg-primary/5 shadow-glow"
            : "border-border hover:border-primary/50 bg-card"
        }`}
      >
        <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center mb-4">
          <Upload className="h-7 w-7 text-primary" />
        </div>
        <p className="text-foreground font-medium mb-1">Drag & drop your file here</p>
        <p className="text-sm text-muted-foreground mb-3">or click to browse</p>
        <div className="flex gap-2">
          <span className="text-xs bg-secondary text-secondary-foreground px-2 py-1 rounded">JPG</span>
          <span className="text-xs bg-secondary text-secondary-foreground px-2 py-1 rounded">PNG</span>
          <span className="text-xs bg-secondary text-secondary-foreground px-2 py-1 rounded">MP4</span>
        </div>
        <p className="text-xs text-muted-foreground mt-2">Max file size: 10MB</p>
        <input
          type="file"
          accept=".jpg,.jpeg,.png,.mp4"
          className="hidden"
          onChange={(e) => e.target.files?.[0] && validateAndSet(e.target.files[0])}
        />
      </label>
      {error && (
        <p className="mt-2 text-sm text-destructive">{error}</p>
      )}
    </div>
  );
};

export default FileUpload;
