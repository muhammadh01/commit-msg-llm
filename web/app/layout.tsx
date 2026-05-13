import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "AI Commit Messages",
  description: "Fine-tuned Qwen2.5-0.5B that generates clean commit messages from git diffs. Full MLOps stack on Kubernetes.",
  openGraph: {
    title: "AI Commit Messages",
    description: "AI-generated commit messages, self-hosted on Kubernetes.",
    url: "https://commit-msg.durak.dev",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        {children}
      </body>
    </html>
  );
}
