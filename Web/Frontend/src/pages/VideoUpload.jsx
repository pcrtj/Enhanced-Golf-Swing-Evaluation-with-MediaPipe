import { useState, useRef, useEffect } from 'react';
import { Link, useNavigate } from "react-router-dom";
import '../css/videoupload.css';
import BoxReveal from "../components/magicui/box-reveal";
import axios from 'axios';

function UploadVideo() {
  const [videoPreview, setVideoPreview] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [showPopup, setShowPopup] = useState(true);
  const [startTime, setStartTime] = useState(0);
  const [endTime, setEndTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [processedResult, setProcessedResult] = useState(null);
  const [outputVideoPath, setOutputVideoPath] = useState(null);
  const videoRef = useRef(null);
  const navigate = useNavigate();

  useEffect(() => {
    const username = sessionStorage.getItem('username');
    if (!username) {
      navigate('/login');
    }
  }, [navigate]);

  const handleClosePopup = () => {
    setShowPopup(false);
  };

  const handleVideoUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const videoUrl = URL.createObjectURL(file);
      setVideoPreview(videoUrl);
      setSelectedFile(file);
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
      setEndTime(videoRef.current.duration);
    }
  };

  const handleStartTimeChange = (e) => {
    setStartTime(Number(e.target.value));
    if (videoRef.current) {
      videoRef.current.currentTime = Number(e.target.value);
    }
  };

  const handleEndTimeChange = (e) => {
    setEndTime(Number(e.target.value));
  };

  const handleStart = async () => {
    console.log('Start button clicked');
    if (selectedFile) {
      const formData = new FormData();
      formData.append('video', selectedFile);
      formData.append('startTime', startTime);
      formData.append('endTime', endTime);

      const username = sessionStorage.getItem('username');
      if (!username) {
        console.error('User not logged in');
        navigate('/login');
        return;
      }
      formData.append('username', username);

      try {
        const response = await axios.post('http://localhost:3000/upload', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
        console.log('Response:', response.data);
        setProcessedResult(response.data);
        setOutputVideoPath(response.data.outputVideoPath);
      } catch (error) {
        console.error('Error uploading the video:', error);
      }
    }
  };

  return (
    <div className="page-body">
      {showPopup && (
        <div className='popup-overlay'>
          <div className='popup-content'>
              <h2 className='popup-header'><div className='beforeusing'>Before using the model</div></h2>
              <p className='info-text'>GOLF SWING SEQUENCES : There are 8 golf swing sequences.</p>
              <div className="golf-info-container">
                <div className="golf-image">
                  <img src="../images/golfswing.png" alt="Golf Swing Sequences" />
                </div>
                <div className="golf-sequences">
                  <ul className="golf-swing-sequences">
                    <p className='golfswingtext'>
                      <div className='infotextgreen'>ADDRESS (A)</div>
                    </p>
                    <p className='golfswingtext'>
                      <div className='infotextgreen'>TOE-UP (TU)</div>
                    </p>
                    <p className='golfswingtext'>
                      <div className='infotextgreen'>MID-BACKSWING (MB)</div>
                    </p>
                    <p className='golfswingtext'>
                      <div className='infotextgreen'>TOP (T)</div>
                    </p>
                    <p className='golfswingtext'>
                      <div className='infotextgreen'>MID-DOWNSWING (MD)</div>
                    </p>
                    <p className='golfswingtext'>
                      <div className='infotextgreen'>IMPACT (I)</div>
                    </p>
                    <p className='golfswingtext'>
                      <div className='infotextgreen'>MID-FOLLOW-THROUGH (MFT)</div>
                    </p>
                    <p className='golfswingtext'>
                      <div className='infotextgreen'>FINISH (F)</div>
                    </p>
                  </ul>
                </div>
              </div>
              <div className='footcontent'>
                Please make sure your video <div className='subtextgreen'>starts with the Address</div> and <div className='subtextred'>ends with the Finish</div>.
              </div>
              <button onClick={handleClosePopup} className='close-button'>
                continue
              </button>
            </div>
        </div>
      )}
      {!showPopup && (
        <div>
          <div className="upload-container">
            <BoxReveal boxColor={"rgba(64, 83, 76, 0.8)"} duration={0.5}>
              <h1 className='headerhome'>Upload Your Video</h1>
              <p className='content'>
                Please make sure your video <span className='subtextgreen'>starts with the Address</span> and <span className='subtextred'>ends with the Finish</span>.
              </p>
            </BoxReveal>
            <div className="upload-input">
              <input type="file" accept="video/*" onChange={handleVideoUpload} />
              {videoPreview ? (
                <div className="video-preview-container">
                  <video 
                    ref={videoRef}
                    src={videoPreview} 
                    controls 
                    onLoadedMetadata={handleLoadedMetadata}
                    style={{ width: '100%', height: '100%' }} 
                  />
                  <div className="trim-controls">
                    <input 
                      type="range" 
                      min="0" 
                      max={duration} 
                      value={startTime}
                      onChange={handleStartTimeChange}
                    />
                    <input 
                      type="range" 
                      min="0" 
                      max={duration} 
                      value={endTime}
                      onChange={handleEndTimeChange}
                    />
                  </div>
                  <div className="trim-times">
                    <span>Start: {startTime.toFixed(2)}s</span>
                    <span>End: {endTime.toFixed(2)}s</span>
                  </div>
                </div>
              ) : (
                <p className='click'>Click to upload video</p>
              )}
            </div>
            {videoPreview ? (
              <div className="button-container">
                <button className="back-button" onClick={() => {
                  setVideoPreview(null);
                  setSelectedFile(null);
                  setStartTime(0);
                  setEndTime(0);
                }}>CANCEL</button>
                <button className="start-button" onClick={handleStart}>START</button>
              </div>
            ) : (
              <div className="button-container">
                <Link to="/homeuser">
                  <button className="back-button">BACK</button>
                </Link>
              </div>
            )}
          </div>
          {processedResult && (
            <div className="result-container">
              <h2>Processing Result</h2>
              <p>Average Accuracy: {processedResult.avgAccuracy}%</p>
              <p>Accuracy per Phase:</p>
              <ul>
                {processedResult.accuracy.map((acc, index) => (
                  <li key={index}>{`Phase ${index + 1}: ${acc}%`}</li>
                ))}
              </ul>
              {outputVideoPath && (
                <video controls src={`/uploads/${outputVideoPath.split('/').slice(-3).join('/')}`} style={{ width: '100%', maxWidth: '500px' }} />
              )}
              <button onClick={() => navigate('/history')}>View History</button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default UploadVideo;