import '../css/HomePage.css';
import { useEffect } from 'react';
import LenisScroll from './LeninScroll';


const HomePage = () => {
  useEffect(() => {
    document.body.style.overflow = "auto";
}, []);

  return (
    <div>
      <LenisScroll />
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
  <div className="objective">
    <h2 className="section-title-objective">Objective</h2>
    <p className='subtitle-objective'>Describe the main goal of the project.</p>

    <section className="min-h-screen bg-gray-900 text-center py-20 px-8 xl:px-0 flex flex-col justify-center">
      <div className="grid-offer text-left grid sm:grid-cols-4 md:grid-cols-2 gap-5 max-w-5xl mx-auto">
        <div className="card bg-gray-800 p-10 relative">
          <div className="circle">
          </div>
          <div className="relative lg:pr-52">
            <h2 className="capitalize text-white mb-4 text-2xl xl:text-3xl">uI/uX <br /> creative design</h2>
            <p className="text-gray-400">Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Pellentesque habitant morbi tristique senectus et netus et malesuada fames.</p>
          </div>
          <div className="icon"></div>
        </div>
        <div className="card bg-gray-800 p-10 relative">
          <div className="circle">
          </div>
          <div className="relative lg:pl-48">
            <h2 className="capitalize text-white mb-4 text-2xl xl:text-3xl">visual <br /> graphic design</h2>
            <p className="text-gray-400">Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Pellentesque habitant morbi tristique senectus et netus et malesuada fames.</p>
          </div>
        </div>
        <div className="card bg-gray-800 p-10 relative">
          <div className="circle">
          </div>
          <div className="relative lg:pr-44">
            <h2 className="capitalize text-white mb-4 text-2xl xl:text-3xl">strategy & <br />digital marketing</h2>
            <p className="text-gray-400">Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Pellentesque habitant morbi tristique senectus et netus et malesuada fames.</p>
          </div>
        </div>
        <div className="card bg-gray-800 p-10 relative">
          <div className="circle">
          </div>
          <div className="relative lg:pl-48">
            <h2 className="capitalize text-white mb-4 text-2xl xl:text-3xl">effective<br /> business growth</h2>
            <p className="text-gray-400">Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Pellentesque habitant morbi tristique senectus et netus et malesuada fames.</p>
          </div>
        </div>
      </div>

    </section>
  </div>
</section>


    <section className="section5">
      <h2 className="section-title">bra bra bra!!!</h2>
      <p className="section-text">
        The primary objective of this project is to capture the breathtaking beauty of nature through artistic expressions and scientific understanding, promoting awareness and conservation efforts to sustain the environment for future generations.
      </p>
    </section>
    </div>
    
    
  );
};

export default HomePage;
