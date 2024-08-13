import Axios from "axios";
import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "../css/login.css";

export default function Sign_up() {
  const [username, setusername] = useState("");
  const [password, setpassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const navigate = useNavigate();

  document.body.style.overflow = "hidden";

  const requst_login = () => [
    Axios.post("http://localhost/login", {
      username: username,
      password: password,
    }).then((Response) => {
      console.log(Response.data)
      if (Response.data == "Success") {
        sessionStorage.setItem("username", username)
        console.log(sessionStorage.getItem("username"))
        navigate('/home');
      } else {
        sessionStorage.setItem("usernamelogin", "null");
        sessionStorage.setItem("login_status", "false");
        sessionStorage.setItem("role", "null");
      }
    }),
  ];

  useEffect(() => {
    console.clear();
  }, []);

  const handleSignUp = (e) => {
    e.preventDefault();
    if (password !== confirmPassword) {
      alert("Passwords do not match");
    } else {
      // Call API to create new user
      Axios.post("http://localhost/signup", {
        username: username,
        password: password,
      }).then((Response) => {
        console.log(Response.data)
        if (Response.data == "Success") {
          alert("User created successfully");
        } else {
          alert("Error creating user");
        }
      });
    }
  };

  return (
    <div className="main">
      <input type="checkbox" id="chk" aria-hidden="true" />
      <div className="signup">
        <form onSubmit={handleSignUp}>
          <label htmlFor="chk" aria-hidden="true">Sign up</label>
          <input type="text" name="username" placeholder="Username" required onChange={(e) => setusername(e.target.value)} />
          <input type="password" name="pswd" placeholder="Password" required onChange={(e) => setpassword(e.target.value)} />
          <input type="password" name="confirmPswd" placeholder="Confirm Password" required onChange={(e) => setConfirmPassword(e.target.value)} />
          <button>Sign up</button>
        </form>
      </div>
      <div className="login">
        <form>
          <label htmlFor="chk" aria-hidden="true">Login</label>
          <input type="text" name="username" placeholder="Username" required onChange={(e) => setusername(e.target.value)} />
          <input type="password" name="pswd" placeholder="Password" required onChange={(e) => setpassword(e.target.value)} />
          <button onClick={requst_login}>Login</button>
        </form>
      </div>
    </div>
  );
}