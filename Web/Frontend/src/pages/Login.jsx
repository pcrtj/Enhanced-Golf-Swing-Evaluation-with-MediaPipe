import Axios from "axios";
import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "../css/login.css";
import Swal from 'sweetalert2'

export default function Sign_up() {
  const [username, setusername] = useState("");
  const [password, setpassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const navigate = useNavigate();

  document.body.style.overflow = "hidden";

  const request_login = () => {

    if (!username || !password ) {
        Swal.fire({
            icon: "warning",
            title: "Warning...",
            text: "Please fill in all fields.",
            timer: 1200,
            showConfirmButton: false
        });
        return;
    }

    Axios.post("http://localhost:3000/login", {
        username: username,
        password: password,
    }).then((Response) => {
        console.log(Response.data);
        if (Response.data === "Success") {
            Swal.fire({
                icon: "success",
                title: "Success...",
                text: "Login successfully!",
                timer: 1200,
                showConfirmButton: false
            }).then(() => {
                sessionStorage.setItem("username",username)
                console.log(sessionStorage.getItem("username"))
                navigate('/');
            });
        } else {
            Swal.fire({
                icon: "error",
                title: "Oops...",
                text: "The username or password is incorrect.",
                timer: 1200,
                showConfirmButton: false
            }).then(() => {
                sessionStorage.setItem("usernamelogin","null");
                sessionStorage.setItem("login_status","false");
                sessionStorage.setItem("role","null");
            });
        }
    }).catch((error) => {
        console.error("Error:", error);
        Swal.fire({
            icon: "error",
            title: "Oops...",
            text: "An error occurred during login. Please try again.",
            timer: 1200,
            showConfirmButton: false
        });
    });
};


  useEffect(() => {
    console.clear();
  }, []);

  
  const request_signup = (e) => {
    e.preventDefault();

    if (!username || !password || !confirmPassword) {
        Swal.fire({
            icon: "warning",
            title: "Warning...",
            text: "Please fill in all fields.",
            timer: 1200,
            showConfirmButton: false
        });
        return;
    }

    if (password !== confirmPassword) {
        Swal.fire({
            icon: "error",
            title: "Oops...",
            text: "Password doesn't match!",
            timer: 1200,
            showConfirmButton: false
        });
        return;
    }

    Axios.post("http://localhost:3000/signup", {
        username: username,
        password: password,
    }).then((Response) => {
        console.log(Response.data);
        if (Response.data == "Success") {
            Swal.fire({
                icon: "success",
                title: "Success...",
                text: "User created successfully!",
                timer: 1200,
                showConfirmButton: false
            });
            // navigate('/login');
        } else if (Response.data == "user already exist") {
            Swal.fire({
                icon: "warning",
                title: "Warning...",
                text: "User already exists!",
                timer: 1200,
                showConfirmButton: false
            });
        } else {
            Swal.fire({
                icon: "error",
                title: "Opps...",
                text: "User created unsuccessfully!",
                timer: 1200,
                showConfirmButton: false
            });
        }
    }).catch((error) => {
        console.error("Error:", error);
        Swal.fire({
            icon: "error",
            title: "Oops...",
            text: "An error occurred during signup. Please try again.",
            timer: 1200,
            showConfirmButton: false
        });
    });
};


  return (
    <body className="login-body">
        <div className="main">
            <input type="checkbox" id="chk" aria-hidden="true" />
                <div className="signup">
                    <form>
                    <label htmlFor="chk" aria-hidden="true">Sign up</label>
                    <input type="text" placeholder="Username"  onChange={(e) => setusername(e.target.value)} />
                    <input type="password" placeholder="Password"  onChange={(e) => setpassword(e.target.value)} />
                    <input type="password" placeholder="Confirm Password"  onChange={(e) => setConfirmPassword(e.target.value)} />
                    <button onClick={request_signup} >Sign up</button>
                    </form>
                </div>
                <div className="login">
                    <form>
                    <label htmlFor="chk" aria-hidden="true">Login</label>
                    <input type="text" placeholder="Username"  onChange={(e) => setusername(e.target.value)} />
                    <input type="password" placeholder="Password"  onChange={(e) => setpassword(e.target.value)} />
                    <button onClick={request_login}>Login</button>
                    </form>
                </div>
        
        </div>
    </body>
  );
}