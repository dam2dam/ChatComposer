import { serialize } from "cookie";

export const setTokenCookie = (res, token) => {
  const cookieOptions = {
    httpOnly: true,
    expires: new Date(
      String(Date.now()) + process.env.COOKIE_EXPIRATION * 1000,
    ), // COOKIE_EXPIRATION은 env 파일에서 설정합니다.
    sameSite: "strict",
    secure: process.env.NODE_ENV === "production", // HTTPS가 아닌 환경에서는 쿠키를 사용하지 않습니다.
    path: "/",
  };

  const serializedToken = serialize("token", token, cookieOptions);

  res.setHeader("Set-Cookie", serializedToken);
};
