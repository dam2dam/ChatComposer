import NextAuth from 'next-auth';
import GoogleProvider from 'next-auth/providers/google';

export default NextAuth({
  /* 사용할 서비스 제공자 설정.
   * 소셜 로그인 추가할 때 이곳에서 설정을 추가한다. */
  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID || "",
      clientSecret: process.env.GOOGLE_SECRET || "",
    }),
  ],

  // http://localhost:3000/api/auth/callback/google

  /* jwt 사용을 위한 임의의 난수를 할당 */
  secret: process.env.SECRET,

  /* 세션 전략을 jwt로 설정 */
  session: {
    strategy: 'jwt',
  },
});