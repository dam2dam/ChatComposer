# 개발환경 설정

## 보일러 플레이트

`yarn create next-app --example with-styled-components with-styled-components-app`
해당 보일러 플레이트는 babel이 아닌 SWC를 이용하여 컴파일링하기 때문에 컴파일 속도가 빠르다.

## lint, prettier

`yarn add -D eslint eslint-config-next eslint-config-prettier`

.prettierrc

```
{
  "singleQuote": false,
  "semi": true,
  "useTabs": false,
  "tabWidth": 2,
  "trailingComma": "all"
}
```

package.json 하단에 아래와 같이 추가

```
  },
  "eslintConfig": {
    "extends": [
      "next/core-web-vitals",
      "prettier"
    ]
  }
}
```

## tailwind

https://tailwindcss.com/docs/guides/nextjs

`yarn add -D tailwindcss postcss autoprefixer`
`npx tailwindcss init -p`
init을 통해 tailwind.config.js and postcss.config.js를 생성한다.

`tailwind.config.js`에 아래와 같이 추가한다.

```
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",

    // Or if using `src` directory:
    "./src/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

vs code extensions에 PostCSS Language Support 를 설치한다.

pages\app.css 에 아래와 같이 추가한다.

```
  @tailwind base;
  @tailwind components;
  @tailwind utilities;
```

pages\app.css를 pages_app.tsx에서 불러온다.

```
import "./app.css";
```

## font

public\fonts\NotoSansKR-Bold.woff2 와 같은 경로로 폰트를 넣어준다.
pages\app.css에 아래와 같이 추가한다.

```
  @tailwind base;
  @tailwind components;
  @tailwind utilities;

  @font-face {
  font-family: 'Noto Sans KR';
    font-style: normal;
    font-weight: 400;
  src: url('/fonts/NotoSansKR-Regular.woff2') format("woff2"),
  url('/fonts/NotoSansKR-Regular.otf') format('opentype');
  font-display: swap;
  }
  @font-face {
    font-family: 'Noto Sans KR';
      font-style: normal;
      font-weight: 700;
    src: url('/fonts/NotoSansKR-Bold.woff2') format("woff2"),
    url('/fonts/NotoSansKR-Bold.otf') format('opentype');
    font-display: swap;
  }

  html,
  body {
    font-family: "Noto Sans KR"
  }
```

## recoil, recoil-persist

아래의 깃헙 페이지에서 리코일의 기본적인 사용법이 나와있다.
https://parkgang.github.io/blog/2021/05/06/using-recoil-in-nextjs/
`yarn add recoil`

```
+import { RecoilRoot } from "recoil";

function MyApp({ Component, pageProps }) {
  return (
    +<RecoilRoot>
      <Component {...pageProps} />
    +</RecoilRoot>
  );
}

export default MyApp;
```

## recoil-persist

리코일 퍼시스트의 기본적인 사용법은 npm 공식문서를 참조한다.
https://www.npmjs.com/package/recoil-persist
https://velog.io/@timosean/Web-Recoil-persist-%EC%82%AC%EC%9A%A9%ED%95%B4%EB%B3%B4%EA%B8%B0

`yarn add recoil-persist`

```
+import { recoilPersist } from 'recoil-persist'
+const { persistAtom } = recoilPersist()
const counterState = atom({
  key: 'count',
  default: 0,
+ effects_UNSTABLE: [persistAtom],
})
```

이때 next.js에서는 클라이언트의 상태와 SSR의 상태가 달라 에러가 발생한다.
이를 방지하지 위해서는 useEffect를 이용하여 SSR이 끝난 후에 recoil-persist의 상태를 불러오도록 설정하면 된다.

아래는 recoil에 'SsrCompleted'의 default 상태를 false로 설정한 후,
useEffect를 이용해 SsrCompleted 상태를 true로 바꾸는 패턴이다.
SsrCompleted가 true로 바뀌었을 때에 recoil-persist의 상태가 적용된다.

```ts
import { AtomEffect, atom, useSetRecoilState } from "recoil";
import { recoilPersist } from "recoil-persist";

// next.js에서 사용하기 위해 ssr이 끝났는지를 확인하는 state이다.
// 새로고침 시에 항상 default값인 false를 갖는다.
const ssrCompletedState = atom({
  key: "SsrCompleted",
  default: false,
});

//useEffect에 쓰일 함수를 정의한다.
export const useSsrComplectedState = () => {
  const setSsrCompleted = useSetRecoilState(ssrCompletedState);
  return () => setSsrCompleted(true);
};

const { persistAtom } = recoilPersist();

//ssrCompletedState가 완료될 때까지 기다린 후, persistAtom을 반환한다.
export const persistAtomEffect = <T>(param: Parameters<AtomEffect<T>>[0]) => {
  param.getPromise(ssrCompletedState).then(() => persistAtom(param));
};
```

상태에는 persistAtom대신 persistAtomEffect를 effect로 넣어주면 된다.

```ts
const counterState = atom({
  key: 'count',
  default: 0,
- effects_UNSTABLE: [persistAtom],
+ effects_UNSTABLE: [persistAtomEffect],
})
```

해당 persistAtomEffect를 사용하는 컴포넌트에서는 useEffect로 아래의 구문을 추가해주어야 한다.
이를 통해 ssr이 완료되었을 때에 setSsrCompleted가 true로 바뀐다.

```ts
const setSsrCompleted = useSsrComplectedState();
useEffect(setSsrCompleted, [setSsrCompleted]);
```

----- 여기까지 완료

## tanstack query

https://kir93.tistory.com/entry/NextJS%EC%97%90%EC%84%9C-react-query-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0
`yarn add @tanstack/react-query`

# Example app with styled-components

This example features how you use a different styling solution than [styled-jsx](https://github.com/vercel/styled-jsx) that also supports universal styles. That means we can serve the required styles for the first render within the HTML and then load the rest in the client. In this case we are using [styled-components](https://github.com/styled-components/styled-components).

This example uses the Rust-based [SWC](https://nextjs.org/docs/advanced-features/compiler#styled-components) in Next.js for better performance than Babel.

Currently, only the `ssr` and `displayName` transforms have been implemented. These two transforms are the main requirement for using `styled-components` in Next.js.

## Deploy your own

Deploy the example using [Vercel](https://vercel.com?utm_source=github&utm_medium=readme&utm_campaign=next-example) or preview live with [StackBlitz](https://stackblitz.com/github/vercel/next.js/tree/canary/examples/with-styled-components)

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/git/external?repository-url=https://github.com/vercel/next.js/tree/canary/examples/with-styled-components&project-name=with-styled-components&repository-name=with-styled-components)

## How to use

Execute [`create-next-app`](https://github.com/vercel/next.js/tree/canary/packages/create-next-app) with [npm](https://docs.npmjs.com/cli/init), [Yarn](https://yarnpkg.com/lang/en/docs/cli/create/), or [pnpm](https://pnpm.io) to bootstrap the example:

```bash
npx create-next-app --example with-styled-components with-styled-components-app
```

```bash
yarn create next-app --example with-styled-components with-styled-components-app
```

```bash
pnpm create next-app --example with-styled-components with-styled-components-app
```

Deploy it to the cloud with [Vercel](https://vercel.com/new?utm_source=github&utm_medium=readme&utm_campaign=next-example) ([Documentation](https://nextjs.org/docs/deployment)).

### Try it on CodeSandbox

[Open this example on CodeSandbox](https://codesandbox.io/s/github/vercel/next.js/tree/canary/examples/with-styled-components)

### Notes

When wrapping a [Link](https://nextjs.org/docs/api-reference/next/link) from `next/link` within a styled-component, the [as](https://styled-components.com/docs/api#as-polymorphic-prop) prop provided by `styled` will collide with the Link's `as` prop and cause styled-components to throw an `Invalid tag` error. To avoid this, you can either use the recommended [forwardedAs](https://styled-components.com/docs/api#forwardedas-prop) prop from styled-components or use a different named prop to pass to a `styled` Link.

<details>
<summary>Click to expand workaround example</summary>
<br />

**components/StyledLink.js**

```javascript
import Link from "next/link";
import styled from "styled-components";

const StyledLink = ({ as, children, className, href }) => (
  <Link href={href} as={as} passHref>
    <a className={className}>{children}</a>
  </Link>
);

export default styled(StyledLink)`
  color: #0075e0;
  text-decoration: none;
  transition: all 0.2s ease-in-out;

  &:hover {
    color: #40a9ff;
  }

  &:focus {
    color: #40a9ff;
    outline: none;
    border: 0;
  }
`;
```

**pages/index.js**

```javascript
import StyledLink from "../components/StyledLink";

export default () => (
  <StyledLink href="/post/[pid]" forwardedAs="/post/abc">
    First post
  </StyledLink>
);
```

</details>
