package com.a504.chatcomposer.global.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.SecurityBuilder;
import org.springframework.security.config.annotation.method.configuration.EnableMethodSecurity;
import org.springframework.security.config.annotation.web.WebSecurityConfigurer;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.builders.WebSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configuration.WebSecurityConfigurerAdapter;
import org.springframework.security.config.annotation.web.configuration.WebSecurityCustomizer;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.CorsConfigurationSource;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;

import lombok.RequiredArgsConstructor;

import java.util.Arrays;

@Configuration
@EnableWebSecurity
@EnableMethodSecurity(
        prePostEnabled = true,
        securedEnabled = true,
        jsr250Enabled = true
)
@RequiredArgsConstructor
public class SecurityConfig {

//    private final CustomOAuth2UserService customOAuth2UserService;
//    private final OAuth2LoginSuccessHandler oAuth2LoginSuccessHandler;
//    private final OAuth2LoginFailureHandler oAuth2LoginFailureHandler;
//    private final JwtAuthenticationEntryPoint jwtAuthenticationEntryPoint;
//    private final JwtAccessDeniedHandler jwtAccessDeniedHandler;
//    private final JwtTokenProvider jwtTokenProvider;
//    private final RedisUtil redisUtil;

    private static final String[] PERMIT_URL_ARRAY = {
            /* swagger v3 */
            "/v3/api-docs/**", //이건 안쓰임;;
            "/swagger*/**",
            "/api-docs/**", //이게 쓰임
            /* swagger v2 */
            "/v2/api-docs",
            "/swagger-resources",
            "/swagger-resources/**",
            "/configuration/ui",
            "/configuration/security",
            "/swagger-ui.html",
            "/webjars/**"
    };

    @Bean
    protected SecurityFilterChain securityFilterChain(HttpSecurity httpSecurity) throws Exception {

        httpSecurity
                // rest api 이므로 Http basic Auth 기반 로그인 인증 해제
                .httpBasic().disable()
                // token 방식이므로 불필요한 csrf 보안 설정 해제
                .csrf().disable()
                // CORS 설정
                .cors().configurationSource(corsConfigurationSource())

                .and()    // token 방식이므로 stateless 설정
                .sessionManagement()
                .sessionCreationPolicy(SessionCreationPolicy.STATELESS)




                .and()    // 요청 권한 설정
                .authorizeRequests()
                .anyRequest().permitAll()
//                .antMatchers(PERMIT_URL_ARRAY).permitAll()
//                .antMatchers("/peak/**").permitAll()
//                .antMatchers("/idol/**").permitAll()
//                .antMatchers("/news/**").permitAll()
//                .antMatchers("/youtube/**").permitAll()
//
//                .anyRequest().authenticated()    // 이외의 모든 요청은 인증 필요
                // .antMatchers("/user/**").hasAnyRole("USER", "GUEST")
                // .anyRequest().permitAll()

//                .and()    // OAuth2 로그인 설정
//                .oauth2Login()
//                .userInfoEndpoint().userService(customOAuth2UserService)
//                .and()
//                .successHandler(oAuth2LoginSuccessHandler)
//                .failureHandler(oAuth2LoginFailureHandler)
//
//                .and()    // 예외 처리 설정
//                .exceptionHandling()
//                // 유효한 자격증명을 제공하지 않고 접근하려 할 때 401 Unauthorized 에러 리턴
//                .authenticationEntryPoint(jwtAuthenticationEntryPoint)
//                // 필요한 권한이 존재하지 않는 경우에 403 Forbidden 에러 리턴
//                .accessDeniedHandler(jwtAccessDeniedHandler)
//
//                .and()    // 필터 체인에 커스텀 필터 추가 설정
//                // JwtAuthenticationFilter를 (UsernamePasswordAuthenticationFilter 전에) 걸어줌
//                .addFilterBefore(new JwtAuthenticationFilter(jwtTokenProvider, redisUtil),
//                        UsernamePasswordAuthenticationFilter.class)
                ;

        return httpSecurity.build();
    }

    public WebSecurityCustomizer webSecurityCustomizer() {
        return (web) -> web.ignoring()
                .antMatchers();
    }

    @Bean
    public CorsConfigurationSource corsConfigurationSource() {

        CorsConfiguration configuration = new CorsConfiguration();
        configuration.addAllowedOrigin("http://localhost:3000");
        configuration.addAllowedOrigin("https://k8a504.p.ssafy.io");
//        configuration.addAllowedOrigin("https://accounts.kakao.com");
        configuration.addAllowedHeader("*");
        configuration.addAllowedMethod("*");
        configuration.setExposedHeaders(Arrays.asList("x-auth-token"));
        configuration.setAllowCredentials(true);

        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/**", configuration);

        return source;
    }


}