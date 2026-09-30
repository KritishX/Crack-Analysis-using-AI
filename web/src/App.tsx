import { useEffect } from "react";
import { Analyzer } from "./components/Analyzer";
import { Footer } from "./components/Footer";
import { Hero } from "./components/Hero";
import { Model } from "./components/Model";
import { Nav } from "./components/Nav";
import { Pipeline } from "./components/Pipeline";
import { ScrollTrigger } from "./lib/motion";

export default function App() {
  // Web fonts change text metrics; re-measure scroll positions once they land.
  useEffect(() => {
    document.fonts.ready.then(() => ScrollTrigger.refresh());
  }, []);

  return (
    <>
      <Nav />
      <main>
        <Hero />
        <Analyzer />
        <Pipeline />
        <Model />
      </main>
      <Footer />
    </>
  );
}
