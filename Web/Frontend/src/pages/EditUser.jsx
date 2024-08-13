import Axios from 'axios';
import { useEffect, useState } from 'react';
import { useNavigate } from "react-router-dom";
// import '../index.css';
import Swal from 'sweetalert2';
import '../css/edit.css';

function EditUser() {
    const [user, setUser] = useState({ U_Firstname: '', U_Lastname: '', U_Gender: '', U_Birthday: '' }); // ปรับปรุงให้เก็บข้อมูลเป็น object
    const navigate = useNavigate();
    useEffect(() => {
        Axios.post("http://192.168.15.227:6956/getUser", {
            UID: sessionStorage.getItem("editID")
        })
            .then((response) => {
                setUser(response.data[0]); // อัปเดต state ด้วยข้อมูลที่ได้
            })
            .catch((error) => {
                console.error("Error fetching user data:", error);
            });
    }, []);

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setUser({ ...user, [name]: value }); // อัปเดตข้อมูลใน state ตามการเปลี่ยนแปลงของ input
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        Axios.post('http://192.168.15.227:6956/updateUser', {
            ...user, // This spread operator passes all user fields
            UID: sessionStorage.getItem("editID") // Ensure you're sending the UID to the server
        })
        .then((response) => {
            console.log(response.status)
            if(response.status==200){
                Swal.fire({
                    title: "All Done!",
                    text: "Updated Informations",
                    icon: "success"
                  });
                navigate('/manage');
            }
            // console.log('User updated successfully:', response);
            // 
            // Handle success, perhaps redirect or display a success message
        })
        .catch((error) => {
            console.error("Error updating user:", error);
            // Handle error, display error message to user
        });

    };
    
    const deLete = async (UID) => {
        if (confirm("Delete User " + UID + "?") == true) {
          try {
            const response = await Axios.post('http://192.168.15.227:6956/changeStatus', { uid: UID });
            peopleList()
          } catch (error) {
            console.error('Error during axios request:', error);
          }
        } else {
          alert("Cancel");
          navigate('/manage');
        }
      }
    

    return (
        
        <div>
            <div className='topadd-edit'>Edit Info</div>
            <div className='editbackground'>
            <div className='container-edit'>
                <form onSubmit={handleSubmit} className='box'>
                    <div className='editlist'>
                        <div className='textF'>Firstname</div>
                        <input className='boxF' type="text" name="U_Firstname" value={user.U_Firstname} onChange={handleInputChange} />
                        <div className='textL'>Lastname</div>
                        <input className='boxL' type="text" name="U_Lastname" value={user.U_Lastname} onChange={handleInputChange}  />
                        <div className='textF'>Gender</div>
                        <select className='editgender' name="U_Gender" value={user.U_Gender} onChange={handleInputChange} >
                            <option value="Male">Male</option>
                            <option value="Female">Female</option>
                        </select>
                        <div className='textF'>Birthday</div>
                        <input className='editdate' type="date" name="U_Birthday" value={user.U_Birthday} onChange={handleInputChange}  />
                    
                        <button className='butsave' type="submit" >Update</button>
                        <button className='butdelete' onClick={()=>{deLete(sessionStorage.getItem("editID"))}}>Delete</button>
                    </div>
                   
                </form>

            
            </div>
        </div>
        </div>
        


    );
}

export default EditUser;
