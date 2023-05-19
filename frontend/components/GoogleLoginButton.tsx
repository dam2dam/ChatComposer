import { useGoogleLogin } from "react-google-login";
import { useEffect, useState } from "react";
import ModalWrapper from "../src/components/shared/ModalWrapper/ModalWrapper";

const GoogleLoginButton = () => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  // const GOOGLE_CLIENT_ID = "339222316329-r328jilv2je6g4n2hofl5f97gr8nu3r6.apps.googleusercontent.com"
  const clientId = process.env.GOOGLE_CLIENT_ID || "";

  const onSuccess = (res: any) => {
    // Handle successful login
  };

  const onFailure = (res: any) => {
    // Handle failed login
  };

  const { signIn } = useGoogleLogin({
    onSuccess,
    onFailure,
    clientId,
    isSignedIn: false,
  });

  useEffect(() => {
    if (isModalOpen) {
      signIn();
    }
  }, [isModalOpen, signIn]);

  const handleButtonClick = () => {
    setIsModalOpen(true);
  };

  const handleClose = () => {
    setIsModalOpen(false);
  };

  return (
    <>
      <button onClick={handleButtonClick}>Google 로그인</button>
      {isModalOpen && (
        // @ts-ignore
        <ModalWrapper onClose={handleClose}>
          <p>모달 내용</p>
        </ModalWrapper>
      )}
    </>
  );
};

export default GoogleLoginButton;
