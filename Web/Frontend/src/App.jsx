import './App.css';
import { HashRouter, Routes, Route } from "react-router-dom";
import Navbar from "./pages/Navbar";
import Navbarguest from "./pages/Navbarguest";
import Homeuser from "./pages/Homeuser";
import Homeguest from "./pages/Homeguest";
import Login from "./pages/Login";
import History from "./pages/History"
import VideoUpload from "./pages/VideoUpload"

function App() {
  return (
    <div className='maindisplay'>
      <HashRouter>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route 
            path="/" 
            element={
              <>
                <Navbarguest />
                <Homeguest />
              </>
            } 
          />
          <Route 
            path="/homeuser" 
            element={
              <>
                <Navbar />
                <Homeuser />
              </>
            } 
          />
          <Route
            path="/history"
            element={
              <>
                <Navbar />
                <History />
              </>
            }
            />
          <Route
            path='/videoupload'
            element={
              <>
                <Navbar />
                <VideoUpload />
              </>
            }
            />
        </Routes>
      </HashRouter>
    </div>
  );
}

export default App;
