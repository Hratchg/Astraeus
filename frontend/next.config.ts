import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  // Required: prevents casing mismatch warnings caused by ~/package-lock.json
  outputFileTracingRoot: path.join(__dirname),
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: false,
  },
};

export default nextConfig;
