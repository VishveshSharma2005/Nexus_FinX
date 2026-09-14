import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FinX — Loan Agreement Intelligence",
  description:
    "Understand your loan agreement: plain-language explanations, RBI-checked clauses, and the real cost in rupees.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
