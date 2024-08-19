import './App.css'
import { HashRouter, Routes, Route } from "react-router-dom";
import Navbar from "./pages/Navbar";
import Home from "./pages/Home";
// import Footer from "./pages/Footer";
import Login from "./pages/Login";

function App() {
  return (
    <div className='maindisplay'>
      <HashRouter>
        <Routes>
          {/* Specify routes without Navbar */}
          {/* <Route path="/" element={<Login />} /> */}
          <Route path="/login" element={<Login />} />

          {/* Specify routes with Navbar */}
          <Route 
            path="/*"
            element={
              <>
                <Navbar />
                <Routes>
                  <Route path="/" element={<Home />} />
                  <Route path="/home" element={<Home />} />
                </Routes>
                {/* <Footer /> */}
              </>
            }
          />
        </Routes>

      </HashRouter>

    </div>
  );
}

export default App;