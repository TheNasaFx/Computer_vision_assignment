import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Magic Mirror — Air Drums + AR Filters",
  description:
    "Pose-driven web mirror. Predictive air-drum engine plus 468-landmark MediaPipe FaceMesh AR filters, all in the browser.",
  keywords: [
    "pose estimation", "MediaPipe", "FaceMesh",
    "Web Audio", "AR filter", "YOLO pose", "computer vision",
  ],
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
