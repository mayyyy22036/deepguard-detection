import { motion } from "framer-motion";
import { Brain, Database, Server, GitBranch, Shield, BarChart3, Github } from "lucide-react";

const architecture = [
  { label: "Model", value: "EfficientNet-B0" },
  { label: "Framework", value: "PyTorch" },
  { label: "Dataset", value: "FaceForensics++ (3200 images)" },
  { label: "Task", value: "Binary Classification (Real vs Fake)" },
  { label: "Input Size", value: "224 × 224 RGB" },
  { label: "Backend", value: "FastAPI" },
];

const metrics = [
  { label: "Accuracy", value: "99%" },
  { label: "AUC-ROC", value: "0.995" },
  { label: "Precision", value: "98.7%" },
  { label: "Recall", value: "99.2%" },
  { label: "F1-Score", value: "98.9%" },
  { label: "Inference", value: "< 100ms" },
];

const stack = [
  { icon: Brain, name: "PyTorch", desc: "Deep learning framework for model training and inference." },
  { icon: Database, name: "FaceForensics++", desc: "Primary dataset with 3200 real and manipulated face images." },
  { icon: Server, name: "FastAPI", desc: "Production-ready REST API with automatic Swagger documentation." },
  { icon: BarChart3, name: "MLflow", desc: "Experiment tracking, model versioning, and artifact management." },
  { icon: GitBranch, name: "Docker", desc: "Containerized deployment with docker-compose for reproducibility." },
  { icon: Shield, name: "EfficientNet-B0", desc: "Efficient CNN architecture with compound scaling for optimal accuracy." },
];

const About = () => {
  return (
    <div className="min-h-screen py-12">
      <div className="container mx-auto px-4 max-w-4xl">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mb-12">
          <h1 className="text-4xl font-bold mb-4">About DeepGuard</h1>
          <p className="text-muted-foreground leading-relaxed max-w-2xl">
            DeepGuard is a deep learning project for detecting manipulated facial images (deepfakes).
            It uses EfficientNet-B0 trained on FaceForensics++ to classify images as real or fake
            with high accuracy and real-time inference.
          </p>
        </motion.div>

        {/* Architecture */}
        <motion.section
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="mb-12"
        >
          <h2 className="text-2xl font-bold mb-6">Model Architecture</h2>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {architecture.map((item, i) => (
              <motion.div
                key={item.label}
                initial={{ opacity: 0, y: 15 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
                viewport={{ once: true }}
                className="bg-card border border-border rounded-xl p-4"
              >
                <div className="text-xs text-muted-foreground uppercase tracking-wider mb-1">{item.label}</div>
                <div className="text-sm font-semibold text-foreground">{item.value}</div>
              </motion.div>
            ))}
          </div>
        </motion.section>

        {/* Performance */}
        <motion.section
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="mb-12"
        >
          <h2 className="text-2xl font-bold mb-6">Performance Metrics</h2>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {metrics.map((m, i) => (
              <motion.div
                key={m.label}
                initial={{ opacity: 0, y: 15 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
                viewport={{ once: true }}
                className="bg-card border border-border rounded-xl p-4"
              >
                <div className="text-xs text-muted-foreground uppercase tracking-wider mb-1">{m.label}</div>
                <div className="text-xl font-bold text-gradient-primary">{m.value}</div>
              </motion.div>
            ))}
          </div>
        </motion.section>

        {/* Tech Stack */}
        <motion.section
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="mb-12"
        >
          <h2 className="text-2xl font-bold mb-6">Technology Stack</h2>
          <div className="grid sm:grid-cols-2 gap-4">
            {stack.map((s, i) => (
              <motion.div
                key={s.name}
                initial={{ opacity: 0, y: 15 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
                viewport={{ once: true }}
                className="flex gap-4 bg-card border border-border rounded-xl p-4 hover:shadow-glow transition-shadow"
              >
                <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                  <s.icon className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <h3 className="font-semibold text-sm">{s.name}</h3>
                  <p className="text-xs text-muted-foreground mt-0.5 leading-relaxed">{s.desc}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.section>

        {/* GitHub */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="text-center py-8"
        >
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 bg-gradient-primary text-primary-foreground font-semibold px-6 py-3 rounded-xl shadow-glow hover:shadow-glow-lg transition-all"
          >
            <Github className="h-5 w-5" />
            View on GitHub
          </a>
        </motion.div>

        <footer className="border-t border-border pt-6 text-center text-sm text-muted-foreground">
          <p>DeepGuard — Deep Learning Project — Instructor: Haythem Ghazouani</p>
        </footer>
      </div>
    </div>
  );
};

export default About;
