import { Loader2 } from "lucide-react";

const LoadingSpinner = ({ text = "Analyzing..." }: { text?: string }) => (
  <div className="flex flex-col items-center justify-center gap-4 py-12">
    <div className="relative">
      <div className="absolute inset-0 rounded-full bg-primary/20 animate-ping" />
      <Loader2 className="h-12 w-12 text-primary animate-spin relative z-10" />
    </div>
    <p className="text-sm font-medium text-muted-foreground">{text}</p>
  </div>
);

export default LoadingSpinner;
