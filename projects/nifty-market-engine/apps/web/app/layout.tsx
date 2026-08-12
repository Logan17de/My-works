import type { Metadata } from "next";
import "./styles.css";

export const metadata: Metadata = { title: "NIFTY Market Engine", description: "Breakout/reversal signal monitor" };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
