import React, { useState } from "react";
import GoogleLogin from "react-google-login";

const GoogleAuth = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [userData, setUserData] = useState(null);

  const responseGoogle = (response) => {
    //주석 console.log(response);
    setIsLoggedIn(true);
    setUserData(response.profileObj);
  };

  const logout = () => {
    setIsLoggedIn(false);
    setUserData(null);
  };

  return (
    <div>
      {isLoggedIn ? (
        <div>
          {userData && userData.imageUrl ? (
            <img src={userData.imageUrl} alt={userData.name} />
          ) : null}
          <p>Welcome {userData.name}!</p>
          <button onClick={logout}>Logout</button>
        </div>
      ) : (
        <GoogleLogin
          clientId="YOUR_GOOGLE_OAUTH_CLIENT_ID"
          buttonText="Login with Google"
          onSuccess={responseGoogle}
          onFailure={responseGoogle}
          cookiePolicy={"single_host_origin"}
        />
      )}
    </div>
  );
};

export default GoogleAuth;
