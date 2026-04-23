import { motion } from "framer-motion";
import { ShieldCheck, ShieldAlert, Clock, RotateCcw } from "lucide-react";
import { Progress } from "@/components/ui/progress";

interface ResultProps {
  prediction: "Real" | "Fake";
  confidence: number;
  processingTime: number;
  probabilities: { real: number; fake: number };
  onReset: () => void;
}

const ResultDisplay = ({ prediction, confidence, processingTime, probabilities, onReset }: ResultProps) => {
  const isReal = prediction === "Real";

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="border border-border rounded-xl bg-card overflow-hidden"
    >
      {/* Badge */}
      <div className={`p-6 text-center ${isReal ? "bg-success/10" : "bg-destructive/10"}`}>
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ type: "spring", stiffness: 200, delay: 0.2 }}
          className={`inline-flex items-center gap-3 px-6 py-3 rounded-full text-2xl font-bold ${
            isReal
              ? "bg-success text-success-foreground"
              : "bg-destructive text-destructive-foreground"
          }`}
        >
          {isReal ? <ShieldCheck className="h-7 w-7" /> : <ShieldAlert className="h-7 w-7" />}
          {prediction.toUpperCase()}
        </motion.div>
      </div>

      <div className="p-6 space-y-5">
        {/* Confidence */}
        <div>
          <div className="flex justify-between text-sm mb-2">
            <span className="text-muted-foreground">Confidence</span>
            <span className="font-mono font-semibold text-foreground">{confidence.toFixed(1)}%</span>
          </div>
          <Progress
            value={confidence}
            className={`h-3 ${isReal ? "[&>div]:bg-success" : "[&>div]:bg-destructive"}`}
          />
        </div>

        {/* Probabilities */}
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-secondary rounded-lg p-3 text-center">
            <div className="text-xs text-muted-foreground mb-1">Real</div>
            <div className="text-lg font-bold text-success">{probabilities.real.toFixed(1)}%</div>
          </div>
          <div className="bg-secondary rounded-lg p-3 text-center">
            <div className="text-xs text-muted-foreground mb-1">Fake</div>
            <div className="text-lg font-bold text-destructive">{probabilities.fake.toFixed(1)}%</div>
          </div>
        </div>

        {/* Processing time */}
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Clock className="h-4 w-4" />
          Processing time: <span className="font-mono text-foreground">{processingTime.toFixed(2)}s</span>
        </div>

        {/* Reset */}
        <button
          onClick={onReset}
          className="w-full py-3 border border-border rounded-lg text-sm font-medium text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors flex items-center justify-center gap-2"
        >
          <RotateCcw className="h-4 w-4" />
          Analyze Another
        </button>
      </div>
    </motion.div>
  );
};

export default ResultDisplay;
