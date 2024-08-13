import { useState, useEffect } from 'react';
import '../css/HomePage.css';

const HomePage = () => {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      const isScrolled = window.scrollY > 50;
      if (isScrolled !== scrolled) {
        setScrolled(isScrolled);
      }
    };

    window.addEventListener('scroll', handleScroll);

    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, [scrolled]);

  return (
    <div className="home-page">
      <div className="background-image"></div>
      <h1>Welcome to our website!</h1>
      <div className={`scroll-down-section ${scrolled ? 'scrolled' : ''}`}>
        <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed sit amet nulla auctor, vestibulum magna sed, convallis ex.</p>
        <button>Learn More</button>
      </div>
    </div>
  );
};

export default HomePage;