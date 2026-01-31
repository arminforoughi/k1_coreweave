import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "OpenClawdIRL - Self-Improving Vision",
  description: "Real-time object recognition that learns and improves",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
