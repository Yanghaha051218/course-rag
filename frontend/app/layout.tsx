import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "CourseRAG",
  description: "Answers grounded only in your course materials.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

