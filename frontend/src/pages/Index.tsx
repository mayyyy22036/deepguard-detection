import { motion } from "framer-motion";
import { Shield, ArrowRight, Target, ImageIcon, Zap, Brain, Server, Layers } from "lucide-react";
import { Link } from "react-router-dom";
import StatsCard from "@/components/StatsCard";

const stats = [
  { icon: Target, value: "99%", label: "Accuracy" },
  { icon: ImageIcon, value: "3200+", label: "Images Analyzed" },
  { icon: Zap, value: "Real-time", label: "Detection" },
];

const techStack = [
  { icon: Brain, name: "PyTorch", desc: "Deep learning framework powering our model training and inference pipeline." },
  { icon: Layers, name: "EfficientNet", desc: "State-of-the-art CNN architecture optimized for accuracy and efficiency." },
  { icon: Server, name: "FastAPI", desc: "High-performance REST API serving predictions with auto-generated docs." },
];

const Index = () => {
  return (
    <div>
      {/* Hero */}
      <section className="relative min-h-[90vh] flex items-center overflow-hidden">
        <div className="absolute inset-0 bg-gradient-hero" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-primary/5 rounded-full blur-3xl" />

        <div className="container relative z-10 mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="max-w-3xl mx-auto text-center"
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.2 }}
              className="inline-flex items-center gap-2 rounded-full border border-primary/30 bg-primary/5 px-4 py-1.5 text-sm text-primary mb-8"
            >
              <Shield className="h-4 w-4" />
              AI-Powered Deepfake Detection
            </motion.div>

            <h1 className="text-5xl md:text-7xl font-extrabold leading-tight mb-6 tracking-tight">
              Deep<span className="text-gradient-primary">Guard</span>
              <br />
              <span className="text-3xl md:text-5xl font-semibold text-muted-foreground">
                AI Deepfake Detection
              </span>
            </h1>

            <p className="text-lg text-muted-foreground mb-10 max-w-xl mx-auto leading-relaxed">
              Detect manipulated videos and images with state-of-the-art AI. 
              Powered by EfficientNet and trained on FaceForensics++.
            </p>

            <div className="flex flex-wrap justify-center gap-4">
              <Link
                to="/detect"
                className="inline-flex items-center gap-2 bg-gradient-primary text-primary-foreground font-semibold px-8 py-4 rounded-xl shadow-glow hover:shadow-glow-lg transition-all text-lg"
              >
                Try Detection
                <ArrowRight className="h-5 w-5" />
              </Link>
              <Link
                to="/about"
                className="inline-flex items-center gap-2 border border-border bg-card text-foreground font-semibold px-8 py-4 rounded-xl hover:bg-secondary transition-colors text-lg"
              >
                Learn More
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Stats */}
      <section className="py-20 border-y border-border">
        <div className="container mx-auto px-4">
          <div className="grid md:grid-cols-3 gap-6 max-w-4xl mx-auto">
            {stats.map((stat, i) => (
              <StatsCard key={stat.label} {...stat} delay={i * 0.1} />
            ))}
          </div>
        </div>
      </section>

      {/* Tech Stack */}
      <section className="py-24">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold mb-4">Technology Stack</h2>
            <p className="text-muted-foreground max-w-md mx-auto">
              Built with industry-standard tools for reliable, scalable deepfake detection.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-6 max-w-4xl mx-auto">
            {techStack.map((tech, i) => (
              <motion.div
                key={tech.name}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.15 }}
                viewport={{ once: true }}
                className="bg-card border border-border rounded-xl p-6 hover:shadow-glow hover:border-primary/30 transition-all"
              >
                <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                  <tech.icon className="h-6 w-6 text-primary" />
                </div>
                <h3 className="text-xl font-semibold mb-2">{tech.name}</h3>
                <p className="text-muted-foreground text-sm leading-relaxed">{tech.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border py-8">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          <p>DeepGuard — Deep Learning Project — Instructor: Haythem Ghazouani</p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
