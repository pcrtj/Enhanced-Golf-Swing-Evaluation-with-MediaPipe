import { useEffect } from 'react';
import Lenis from 'lenis';

function LenisScroll() {
  useEffect(() => {
    const lenis = new Lenis();

    lenis.on('scroll', (e) => {
        console.log(e);
      });
      
    function raf(time) {
        lenis.raf(time);
        requestAnimationFrame(raf);
    }

    requestAnimationFrame(raf);

    return () => {
      lenis.destroy();
    };
  }, []);

  return null;
}

export default LenisScroll;