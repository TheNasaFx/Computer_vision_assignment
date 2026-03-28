import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "YOLO Real-Time Detection | AI-Powered Object Detection",
  description:
    "Upload videos and detect objects in real-time using YOLO26 deep learning model. Fast, accurate, production-grade computer vision.",
  keywords: ["YOLO", "object detection", "computer vision", "deep learning", "AI"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-mesh min-h-screen">{children}</body>
    </html>
  );
}
