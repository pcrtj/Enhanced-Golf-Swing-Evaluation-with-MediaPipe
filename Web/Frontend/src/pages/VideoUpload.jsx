import { useState, useRef, useEffect } from 'react';
import { Link, useNavigate } from "react-router-dom";
import '../css/videoupload.css';
import BoxReveal from "../components/magicui/box-reveal";
import axios from 'axios';
import { Tooltip } from 'react-tooltip';

function UploadVideo() {
  const [videoPreview, setVideoPreview] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [showPopup, setShowPopup] = useState(true);
  const [processedResult, setProcessedResult] = useState(null);
  const [originalVideoPath, setOriginalVideoPath] = useState(null);
  const [processedVideoPath, setProcessedVideoPath] = useState(null);
  const [similarities, setSimilarities] = useState(null);
  const [averageSimilarity, setAverageSimilarity] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showResultPopup, setShowResultPopup] = useState(false);
  const videoRef = useRef(null);
  const navigate = useNavigate();

  useEffect(() => {
    const username = sessionStorage.getItem('username');
    if (!username) {
      navigate('/login');
    }
  }, [navigate]);

  const resetState = () => {
    setVideoPreview(null);
    setSelectedFile(null);
    setProcessedResult(null);
    setOriginalVideoPath(null);
    setProcessedVideoPath(null);
    setSimilarities(null);
    setAverageSimilarity(null);
    setError(null);
  };

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

  const handleCancel = () => {
    resetState();
  };

  const handleStart = async () => {
    setIsLoading(true);
    setError(null);

    if (!selectedFile) {
      setError('No file selected');
      setIsLoading(false);
      return;
    }

    const formData = new FormData();
    formData.append('video', selectedFile);
    
    const username = sessionStorage.getItem('username');
    formData.append('username', username);

    try {
      console.log('Sending request to Flask server...');
      const flaskResponse = await axios.post('http://127.0.0.1:5000/process_video', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        withCredentials: true,
      });

      console.log('Received response from Flask server:', flaskResponse.data);

      if (flaskResponse.data.error) {
        throw new Error(flaskResponse.data.error);
      }

      setProcessedResult(flaskResponse.data);
      setOriginalVideoPath(`http://localhost:3000/uploads/${flaskResponse.data.E_ID}/${flaskResponse.data.input_video}`);
      setProcessedVideoPath(`http://localhost:3000/uploads/${flaskResponse.data.E_ID}/${flaskResponse.data.output_video}`);
      setSimilarities(flaskResponse.data.similarities);
      setAverageSimilarity(flaskResponse.data.average_similarity);

      console.log('Sending data to Express server...');
      await axios.post('http://localhost:3000/save-result', flaskResponse.data);
      console.log('Data saved successfully');

      setShowResultPopup(true);
    } catch (error) {
      console.error('Error processing the video:', error);
      setError(`Error processing the video: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const ResultPopup = () => {
    if (!processedResult || !showResultPopup) return null;

    const handleClose = () => {
      setShowResultPopup(false);
      resetState();
    };

    const poseNames = ['Address', 'Toe-Up', 'Mid-Backswing', 'Top', 'Mid-Downswing', 'Impact', 'Mid-Follow-Through', 'Finish'];
    const angleNames = [
      'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
      'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee'
    ];

    return (
      <div className="result-popup-overlay">
        <div className="result-popup-content">
          <h2 className='headerresult'>Processing Results</h2>
          <div className="similarity-results">
            <h3 className='resultinfo-header'>Pose Similarities</h3>
            <div className="pose-similarities-container">
              {similarities.map((poseSimilarities, index) => {
                const avgSimilarity = poseSimilarities.reduce((a, b) => a + b, 0) / poseSimilarities.length;
                const tooltipId = `pose-tooltip-${index}`;
                return (
                  <div key={index} className="pose-similarity-item">
                    <p 
                      className='info' 
                      data-tooltip-id={tooltipId}
                    >
                      {poseNames[index]}: {avgSimilarity.toFixed(2)}%
                    </p>
                    <Tooltip 
                      id={tooltipId} 
                      place="right"
                      className="custom-tooltip"
                    >
                      <div className="tooltip-content">
                        {angleNames.map((angle, i) => (
                          <div key={i} className="tooltip-item">
                            <span className="angle-name">{angle}:</span>
                            <span className="angle-value">{poseSimilarities[i].toFixed(2)}%</span>
                          </div>
                        ))}
                      </div>
                    </Tooltip>
                  </div>
                );
              })}
            </div>
            <h4 className='resultinfo-header'>Average Similarity: {averageSimilarity.toFixed(2)}%</h4>
          </div>
          <button onClick={handleClose} className='result-close-button'>
            Close
          </button>
        </div>
      </div>
    );
  };

  const LoadingOverlay = () => {
    if (!isLoading) return null;

    return (
      <div className="loading-overlay">
        <div className="wrapper">
          <div className="circle"></div>
          <div className="circle"></div>
          <div className="circle"></div>
          <div className="shadow"></div>
          <div className="shadow"></div>
          <div className="shadow"></div>
        </div>
      </div>
    );
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
              Please make sure your video <div className='subtextgreen'>start with the Address</div> and <div className='subtextred'>end with the Finish</div>.
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
                Please make sure your video <span className='subtextgreen'>start with the Address</span> and <span className='subtextred'>end with the Finish</span>.
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
                    className="video-preview"
                  />
                </div>
              ) : (
                <p className='click'>Click to upload video</p>
              )}
            </div>
            {videoPreview ? (
              <div className="button-container">
                <button className="back-button" onClick={handleCancel}>CANCEL</button>
                <button className="start-button" onClick={handleStart} disabled={isLoading}>
                  {isLoading ? 'PROCESSING...' : 'START'}
                </button>
              </div>
            ) : (
              <div className="button-container">
                <Link to="/homeuser">
                  <button className="back-button">BACK</button>
                </Link>
              </div>
            )}
          </div>
          {error && <p className="error-message">{error}</p>}
        </div>
      )}
      <LoadingOverlay />
      <ResultPopup />
    </div>
  );
}

export default UploadVideo;