import React from 'react';
import '../css/HomePage.css';
import { useEffect } from 'react';



const HomePage = () => {
  useEffect(() => {
    document.body.style.overflow = "auto";
}, []);

  return (
    <div>
      <div className="homepage">
        <div className="content">
          <h1 className='headerhome'>Enhanced Golf Swing Evaluation <mark>with MediaPipe</mark></h1>
            <p className='content'>
              Wireless Ad-Hoc and Sensor Networks Laboratory, <br></br>Department of Computer Science and Information Engineering, <br></br>National Central University, Taiwan
            </p>
            <button className="getstarted">Get Started</button>
        </div>
        <div className="image-placeholder">
          <img className="h-image" src="/images/landscapeBG.jpg" alt="Header-image"></img>
        </div>
      </div>
    
    <section className="section2">
      <div className="content-image-placeholder">
        <img className="content-image" src="/images/Landscape.jpg" alt="content-image"></img>
      </div>
    </section>

    <section className="section3">
    <h2 className="section-title">Introduction</h2>
      <p className="section-text">
        Nature is a vast and diverse realm, encompassing a wide range of ecosystems, landscapes, and living organisms. This project aims to explore the beauty and complexity of the natural world, highlighting its intricate balance and the importance of preserving its delicate equilibrium.
      </p>
    </section>

    <section className="section4">
    <h2 className="section-title">Abstract</h2>
      <p className="section-text">
        Nature is a vast and diverse realm, encompassing a wide range of ecosystems, landscapes, and living organisms. This project aims to explore the beauty and complexity of the natural world, highlighting its intricate balance and the importance of preserving its delicate equilibrium.
      </p>
    </section>

    <section className="section5">
      <h2 className="section-title">Objective</h2>
      <p className="section-text">
        The primary objective of this project is to capture the breathtaking beauty of nature through artistic expressions and scientific understanding, promoting awareness and conservation efforts to sustain the environment for future generations.
      </p>
    </section>
    </div>
    
    
  );
};

export default HomePage;
