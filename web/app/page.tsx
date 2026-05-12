"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { GitCommit, Sparkles, Zap, Code2, BarChart3, ArrowRight, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

const EXAMPLE_DIFFS = [
  { label: "Bug fix", diff: `# MODIFY src/auth.py\n@@ -10,3 +10,5 @@ def login(user):\n     if not user:\n         return None\n+    if user.is_banned:\n+        return None\n     return user.token` },
  { label: "Cache fallback", diff: `# MODIFY src/db.py\n@@ -45,7 +45,7 @@ def get_user(uid):\n-    return cache.get(uid)\n+    return cache.get(uid) or db.query(uid)` },
  { label: "New endpoint", diff: `# ADD src/api/health.py\n+from fastapi import APIRouter\n+router = APIRouter()\n+@router.get("/health")\n+def health():\n+    return {"status": "ok"}` },
];

type GenerateResponse = { message: string; model: string; cached: boolean; latency_ms: number };

export default function Home() {
  const [diff, setDiff] = useState(EXAMPLE_DIFFS[0].diff);
  const [result, setResult] = useState<GenerateResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function generate() {
    setLoading(true); setError(null); setResult(null);
    try {
      const r = await fetch("/api/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ diff }) });
      if (!r.ok) throw new Error(`API ${r.status}`);
      setResult(await r.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : "failed");
    } finally { setLoading(false); }
  }

  return (
    <main className="min-h-screen bg-black text-white antialiased relative overflow-hidden">
      <div className="absolute inset-0 -z-10">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[800px] bg-blue-500/10 rounded-full blur-[120px]" />
        <div className="absolute top-1/3 right-0 w-[600px] h-[600px] bg-purple-500/10 rounded-full blur-[120px]" />
      </div>

      <nav className="flex items-center justify-between px-6 lg:px-12 py-6 border-b border-white/5">
        <div className="flex items-center gap-2">
          <GitCommit className="size-5" />
          <span className="font-mono text-sm">commit-msg-llm</span>
        </div>
        <div className="flex items-center gap-2">
          <a href="https://github.com/muhammadh01/commit-msg-llm" target="_blank" rel="noopener" className="p-2 hover:bg-white/5 rounded-md transition-colors" aria-label="GitHub">
            <Code2 className="size-4" />
          </a>
          <a href="https://commit-grafana.durak.dev" target="_blank" rel="noopener" className="p-2 hover:bg-white/5 rounded-md transition-colors" aria-label="Grafana">
            <BarChart3 className="size-4" />
          </a>
          <a href="https://commit.durak.dev/docs" target="_blank" rel="noopener">
            <Button variant="ghost" size="sm" className="text-white/70 hover:text-white">API docs <ArrowRight className="size-3 ml-1" /></Button>
          </a>
        </div>
      </nav>

      <section className="px-6 lg:px-12 pt-20 pb-12 max-w-5xl mx-auto text-center">
        <Badge variant="secondary" className="mb-6 bg-white/5 text-white/70 border-white/10">
          <Sparkles className="size-3 mr-1.5" /> Fine-tuned Qwen2.5-0.5B
        </Badge>
        <h1 className="text-5xl lg:text-7xl font-medium tracking-tight mb-6 bg-gradient-to-b from-white to-white/40 bg-clip-text text-transparent">
          Commit messages,<br />written by AI
        </h1>
        <p className="text-lg lg:text-xl text-white/50 max-w-2xl mx-auto font-mono">
          Paste a git diff. Get a clean commit message.<br />
          LoRA fine-tuned on 6,676 real commits.
        </p>
      </section>

      <section className="px-6 lg:px-12 pb-20 max-w-5xl mx-auto">
        <div className="grid lg:grid-cols-2 gap-4">
          <div className="rounded-xl border border-white/10 bg-white/[0.02] overflow-hidden">
            <div className="flex items-center justify-between px-4 py-3 border-b border-white/5">
              <span className="text-xs font-mono text-white/40">DIFF</span>
              <div className="flex gap-1">
                {EXAMPLE_DIFFS.map((ex, i) => (
                  <button key={i} onClick={() => setDiff(ex.diff)} className="text-xs font-mono px-2 py-1 rounded hover:bg-white/5 text-white/50 hover:text-white transition-colors">{ex.label}</button>
                ))}
              </div>
            </div>
            <textarea value={diff} onChange={(e) => setDiff(e.target.value)} spellCheck={false} className="w-full h-72 bg-transparent text-sm font-mono p-4 outline-none resize-none text-white/90 placeholder-white/20" placeholder="Paste your diff here..." />
          </div>

          <div className="rounded-xl border border-white/10 bg-white/[0.02] overflow-hidden flex flex-col">
            <div className="flex items-center justify-between px-4 py-3 border-b border-white/5">
              <span className="text-xs font-mono text-white/40">COMMIT MESSAGE</span>
              {result && (
                <div className="flex gap-2 items-center">
                  {result.cached && <Badge variant="secondary" className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20 text-[10px]"><Zap className="size-2.5 mr-1" /> cached</Badge>}
                  <span className="text-xs font-mono text-white/40">{result.latency_ms}ms</span>
                </div>
              )}
            </div>
            <div className="flex-1 p-4 min-h-72 flex items-center justify-center">
              <AnimatePresence mode="wait">
                {loading && (
                  <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex items-center gap-2 text-white/40 text-sm font-mono">
                    <Loader2 className="size-4 animate-spin" /> generating...
                  </motion.div>
                )}
                {!loading && result && (
                  <motion.div key="result" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="w-full">
                    <div className="font-mono text-lg text-white">{result.message}</div>
                    <div className="mt-4 text-xs font-mono text-white/30">{result.model}</div>
                  </motion.div>
                )}
                {!loading && !result && !error && <span className="text-sm font-mono text-white/30">Click generate to see the result</span>}
                {error && <span className="text-sm font-mono text-red-400">{error}</span>}
              </AnimatePresence>
            </div>
          </div>
        </div>

        <div className="flex justify-center mt-6">
          <Button onClick={generate} disabled={loading || !diff.trim()} size="lg" className="bg-white text-black hover:bg-white/90 font-mono">
            {loading ? <><Loader2 className="size-4 mr-2 animate-spin" />Generating</> : <><Sparkles className="size-4 mr-2" />Generate commit message</>}
          </Button>
        </div>
      </section>

      <section className="px-6 lg:px-12 py-20 border-t border-white/5 max-w-5xl mx-auto">
        <h2 className="text-sm font-mono text-white/40 mb-8 text-center">FULL MLOPS STACK</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm font-mono text-white/70">
          {[["TRAINING","Qwen2.5 · LoRA · Kaggle GPU"],["REGISTRY","HuggingFace Hub · MLflow"],["SERVING","FastAPI · Redis"],["INFRA","Kubernetes · DigitalOcean"],["INGRESS","Nginx · Let's Encrypt TLS"],["MONITORING","Prometheus · Grafana"],["CI/CD","GitHub Actions · ghcr.io"],["EVAL","BLEU · ROUGE · GPT-4 judge"]].map(([k,v]) => (
            <div key={k} className="border border-white/5 rounded-lg p-4 hover:bg-white/[0.02] transition-colors">
              <div className="text-white/40 text-[10px] mb-1">{k}</div>
              <div className="text-white">{v}</div>
            </div>
          ))}
        </div>
      </section>

      <footer className="px-6 lg:px-12 py-8 border-t border-white/5 text-center text-xs font-mono text-white/30">
        Built by <a href="https://github.com/muhammadh01" target="_blank" rel="noopener" className="text-white/70 hover:text-white">muhammadh01</a>
      </footer>
    </main>
  );
}
