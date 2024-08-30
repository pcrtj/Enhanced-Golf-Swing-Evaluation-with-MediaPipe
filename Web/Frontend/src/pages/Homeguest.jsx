import '../css/HomePage.css';
import { useEffect } from 'react';
import Swal from 'sweetalert2';
import LenisScroll from './LeninScroll';
import BoxReveal from "../components/magicui/box-reveal";

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
      <LenisScroll />
      <div className="homepage">
        <div className="content">
          <h1 className='headerhome'>
            <BoxReveal boxColor={"rgba(64, 83, 76, 0.8)"} duration={0.5}>
              Enhanced Golf Swing Evaluation <mark>with MediaPipe</mark>
            </BoxReveal>
          </h1>
            <BoxReveal boxColor={"rgba(64, 83, 76, 0.8)"} duration={0.5}>
              <p className='content'>
                Wireless Ad-Hoc and Sensor Networks Laboratory, <br></br>Department of Computer Science and Information Engineering, <br></br>National Central University, Taiwan
              </p>
            </BoxReveal> 
            <button className="getstarted" onClick={handleGetStarted}>Get Start</button>
        </div>
        <div className="image-placeholder">
          <img className="h-image" src="/images/landscapeBG.jpg" alt="Header-image"></img>
        </div>
      </div>
    
      <section className="section2">
        <div className="content-image-placeholder">
          <img className="content-image1" src="/images/Landscape.jpg" alt="content-image"></img>
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
                    <p className='content-intro'>• By reducing the need to rely on trainers or experts who may have <br></br><mark className='introtext'>time and cost constraints</mark>.</p>
                </div>
            </div>
        </div>
      </section>

      <section className="section4">
        <div className="content-image-placeholder">
          <img className="content-image2" src="/images/couplebg.jpg" alt="content-image"></img>
        </div>
      </section>

      <section className="section5">
        <div className="objective">
          <h2 className="section-title-objective">Objective</h2>
          <p className='subtitle-objective'>Describe the main goal of the project.</p>
          <div className="introduction-content">
            <div className="intro-point">
              <p className='content-intro'>
                • To create a model that can evaluate golf swing posture accurately and efficiently using <mark className='introtext'>MediaPipe</mark> technology.
              </p>
            </div>
            <div className="intro-point">
              <p className='content-intro'>
                • To create a model that can <br></br><mark className='introtext'>score golf swings</mark>, <br></br>which will help golfers use as a reference for comparison.
              </p>
            </div>
            <div className="intro-point">
              <p className='content-intro'>
                • To help golfers of all levels practice swing posture more easily, with <mark className='introtext'>correct posture</mark>.
              </p>
            </div>
            <div className="intro-point">
              <p className='content-intro'>
                • To collect data and develop techniques for <mark className='introtext'>further research and development in the future</mark>.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="section6">
        <div className="content-image-placeholder">
          <img className="content-image3" src="/images/golfgolfgolfwth.jpg" alt="content-image"></img>
        </div>
      </section>

      <section className="section7">
      <h2 className="section-title">Thank you</h2>
        <p className="endtext">
          This project is a cooperation between 
          <br>
          </br>
          <p className='endtextncu'>National Central University, Taiwan
          </p> 
          <br>
          </br>and 
          <br>
          </br>
          <p className='endtextkmutnb'>King Mongkut&#39;s University of Technology North Bangkok, Thailand
          </p>.
        </p>
        {/* <img className="icon" src="/images/taiwan.png" alt="taiwan-icon" />
        <img className="icon" src="/images/thailand.png" alt="taiwan-icon" /> */}
      </section>
    </div>
    
  );
};

export default HomeGuest;
