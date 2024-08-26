import React from 'react';
import '../css/HomePage.css';
import { useEffect } from 'react';
import Swal from 'sweetalert2';



const HomeGuest = () => {
  useEffect(() => {
    document.body.style.overflow = "auto";  // เปิดการเลื่อนหน้ากลับมา
}, []);

  const handleGetStarted = () => {
    Swal.fire({
      icon: "info",
      title: "Not available...",
      text: "Please login before using.",
      timer: 1500,
      showConfirmButton: false
    });
  };

  return (
    <div>
      <div className="homepage">
        <div className="content">
          <h1 className='headerhome'>Enhanced Golf Swing Evaluation <mark>with MediaPipe</mark></h1>
            <p className='content'>
              Wireless Ad-Hoc and Sensor Networks Laboratory, <br></br>Department of Computer Science and Information Engineering, <br></br>National Central University, Taiwan
            </p>
            <button className="getstarted" onClick={handleGetStarted}>Get Started</button>
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

    <section id="section3">
      <div className="introduction">
          <h2 className='section-title'>Introduction</h2>
          <p className='subtitle'>A brief overview of the topic.</p>
          <div className="introduction-content">
              <div className="intro-point">
                  <p className='content-intro'>• Golf is a sport that requires <br></br><mark className='introtext'>every part of the body</mark> to work together precisely and with <br></br><mark className='introtext'>perfect coordination</mark>.</p>
              </div>
              <div className="intro-point">
                  <p className='content-intro'>• Develop a model that is capable of <mark className='introtext'>accurately and efficiently</mark> estimating golf swing posture.</p>
              </div>
              <div className="intro-point">
                  <p className='content-intro'>• Practicing the golf swing alone can be challenging without <br></br><mark className='introtext'>understanding correct posture</mark> <br></br>or <br></br><mark className='introtext'>expert guidance.</mark></p>
              </div>
              <div className="intro-point">
                  <p className='content-intro'>• By reducing the need to rely on trainers or experts who may have <mark className='introtext'>time and cost constraints</mark>.</p>
              </div>
          </div>
      </div>
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

export default HomeGuest;
