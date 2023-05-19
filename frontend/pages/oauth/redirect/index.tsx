import { useRecoilState, useSetRecoilState } from "recoil";
import { authTokenState } from "../../../store/atoms";
import { useRouter } from "next/router";
import CssSpinner from "../../../components/cssSpinner";
import { toastAlert } from "../../../utils/toastAlert";
import { useEffect } from "react";

function Oauth() {
  const [authToken, setAuthToken] = useRecoilState(authTokenState);
  const router = useRouter();

  //로컬환경 테스트 주소 http://localhost:3000/oauth/redirect?token=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMDgwNDE3NTg0ODczNzA3MzkwNjkiLCJyb2xlIjoiUk9MRV9VU0VSIiwiZXhwIjoxNjg0MjQzNTM4fQ.bSvre-LYNtmeLQVuXXOy56DVCxEqdhoULDAuHiA5uDU
  const { token } = router.query as { token: string };
  if (token) {
    setAuthToken(() => token);
  }

  useEffect(() => {
    if (!authToken) return;
    toastAlert("로그인 되었습니다.");
    setTimeout(() => router.push("/"), 100);
  }, [authToken]);

  return (
    <div>
      <CssSpinner />
    </div>
  );
}

export default Oauth;
