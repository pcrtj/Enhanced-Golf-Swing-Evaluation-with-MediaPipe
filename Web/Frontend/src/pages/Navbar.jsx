import { useEffect, useState } from "react";
import { BiMenuAltRight } from "react-icons/bi";
import { AiOutlineClose } from "react-icons/ai";
import { Link} from "react-router-dom";
import '../css/Navbar.css'
import Swal from 'sweetalert2'

function Navbar() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [size, setSize] = useState({
    width: 0,
    height: 0,
  });
  useEffect(() => {
    const handleResize = () => {
      setSize({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    };
    window.addEventListener("resize", handleResize);

    return () => window.removeEventListener("resize", handleResize);
  }, []);

  useEffect(() => {
    if (size.width > 768 && menuOpen) {
      setMenuOpen(false);
    }
  }, [size.width, menuOpen]);

  const menuToggleHandler = () => {
    setMenuOpen((p) => !p);
  };

  

  const logout=()=>{
    sessionStorage.setItem("username",'null')
    Swal.fire({
        icon: "success",
        title: "Success...",
        text: "Logout successfully!",
        timer: 1000,
        showConfirmButton: false
  });

  }

  return (
    <header className="header">
      <div className="header__content">
        <div className="header__logo">
          <div className="golflogo">
            <svg fill="#ffffff" width="48px" height="48px" viewBox="-51.2 -51.2 614.40 614.40" xmlns="http://www.w3.org/2000/svg" stroke="#ffffff" transform="matrix(-1, 0, 0, 1, 0, 0)rotate(0)"><g id="SVGRepo_bgCarrier" strokeWidth="0" transform="translate(0,0), scale(1)"><path transform="translate(-51.2, -51.2), scale(19.2)" d="M16,29.2890248090467C17.832956747834555,29.404862415358892,18.898816242932583,26.99272234857879,20.655132745746634,26.45559933435264C22.712707136963243,25.826344551838186,25.425025272731737,27.367226140446665,27.017012423923,25.919762542579818C28.542322768279504,24.532922707061235,28.10776933753788,21.918227636399088,27.901531606511362,19.867042034294006C27.722978056523708,18.091195931272555,25.765051689672806,16.73641056725198,25.91484591699216,14.957907702117526C26.113463106738394,12.599731106463173,29.02759202200969,10.93979639526478,28.852200237931854,8.579778732951077C28.69440069675054,6.456476856235854,27.101833328412184,4.1433349614243,25.06503062975488,3.52305573268797C22.79384596444351,2.8313988169514577,20.590916926794687,4.8901187246947195,18.262687237431386,5.354893494447705C16.78970949205546,5.648937890318965,15.326922570123577,5.768586939318769,13.82488308010141,5.766879447009044C11.943210443759105,5.764740394356089,9.964413617525299,4.5513509899826206,8.257614009672572,5.3435199016040205C6.614521318464638,6.1061208202465895,6.0447636573613215,8.162481373629776,5.205078334003025,9.767549070255715C4.3872065228751325,11.330919999227138,3.087161968260622,12.93280223019675,3.3768005683946924,14.673248277134993C3.680421978637919,16.497717401688938,6.516840334736113,17.172138036134044,6.742813763513942,19.00784213993049C7.045757891252421,21.468820767012,3.3527426850051048,24.074285401319877,4.783362045557071,26.09950617760505C6.109435327937102,27.97672889414689,9.388947852348505,25.254689539574716,11.599662604241193,25.88331960846017C13.403422539540982,26.39622962157224,14.128466542773003,29.17074926540691,16,29.2890248090467" fill="#40534C" strokeWidth="0"></path></g><g id="SVGRepo_tracerCarrier" strokeLinecap="round" strokeLinejoin="round" stroke="#CCCCCC" strokeWidth="5.12"></g><g id="SVGRepo_iconCarrier"><path d="M96 416h224c0 17.7-14.3 32-32 32h-16c-17.7 0-32 14.3-32 32v20c0 6.6-5.4 12-12 12h-40c-6.6 0-12-5.4-12-12v-20c0-17.7-14.3-32-32-32h-16c-17.7 0-32-14.3-32-32zm320-208c0 74.2-39 139.2-97.5 176h-221C39 347.2 0 282.2 0 208 0 93.1 93.1 0 208 0s208 93.1 208 208zm-180.1 43.9c18.3 0 33.1-14.8 33.1-33.1 0-14.4-9.3-26.3-22.1-30.9 9.6 26.8-15.6 51.3-41.9 41.9 4.6 12.8 16.5 22.1 30.9 22.1zm49.1 46.9c0-14.4-9.3-26.3-22.1-30.9 9.6 26.8-15.6 51.3-41.9 41.9 4.6 12.8 16.5 22.1 30.9 22.1 18.3 0 33.1-14.9 33.1-33.1zm64-64c0-14.4-9.3-26.3-22.1-30.9 9.6 26.8-15.6 51.3-41.9 41.9 4.6 12.8 16.5 22.1 30.9 22.1 18.3 0 33.1-14.9 33.1-33.1z"></path></g></svg>
          </div>
          <Link to="/homeuser" className="header__content__logo">
            Golf <span></span>SWING <span></span>Evaluation
          </Link>
        </div>
        <nav className={`${"header__content__nav"} ${menuOpen && size.width < 768 ? `${"isMenu"}` : ""} }`}>
          <ul>
            <li>
              <Link to="/homeuser">
                <button className="loginlogout">
                  Home
                </button>
              </Link>
            </li>
            <li>
              <Link to="/history">
                <button className="loginlogout">
                  History
                </button>
              </Link>
            </li>
            <li>
              <Link to="/">
                <button className="Btn">
                  <div className="sign"><svg viewBox="0 0 512 512"><path d="M377.9 105.9L500.7 228.7c7.2 7.2 11.3 17.1 11.3 27.3s-4.1 20.1-11.3 27.3L377.9 406.1c-6.4 6.4-15 9.9-24 9.9c-18.7 0-33.9-15.2-33.9-33.9l0-62.1-128 0c-17.7 0-32-14.3-32-32l0-64c0-17.7 14.3-32 32-32l128 0 0-62.1c0-18.7 15.2-33.9 33.9-33.9c9 0 17.6 3.6 24 9.9zM160 96L96 96c-17.7 0-32 14.3-32 32l0 256c0 17.7 14.3 32 32 32l64 0c17.7 0 32 14.3 32 32s-14.3 32-32 32l-64 0c-53 0-96-43-96-96L0 128C0 75 43 32 96 32l64 0c17.7 0 32 14.3 32 32s-14.3 32-32 32z"></path></svg></div>
                  <div className="text" onClick={logout} >Logout</div>
                </button>
              </Link>
            </li>
          </ul>
        </nav>
        <div className="header__content__toggle">
          {!menuOpen ? (
            <BiMenuAltRight onClick={menuToggleHandler} />
          ) : (
            <AiOutlineClose onClick={menuToggleHandler} />
          )}
        </div>
      </div>
    </header>
  );
}

export default Navbar;
