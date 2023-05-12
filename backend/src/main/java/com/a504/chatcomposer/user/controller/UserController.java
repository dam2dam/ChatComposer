package com.a504.chatcomposer.user.controller;

import com.a504.chatcomposer.global.common.ApiResponse;
import com.a504.chatcomposer.global.config.properties.AppProperties;
import com.a504.chatcomposer.user.entity.User;
import com.a504.chatcomposer.user.service.UserService;
import com.a504.chatcomposer.oauth.token.AuthToken;
import com.a504.chatcomposer.oauth.token.AuthTokenProvider;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Date;

@RestController
@RequestMapping("/v1/users")
@RequiredArgsConstructor
public class UserController {

    private final String ROLE_USER = "ROLE_USER";
    private final AuthTokenProvider tokenProvider;
    private final UserService userService;
    private final AppProperties appProperties;
    @Value("${jwt.master}")
    private String MASTER_ID;
    @GetMapping
    public ApiResponse getUser() {
        // principal에서 토큰 정보를 받아올 수 있음!
        org.springframework.security.core.userdetails.User principal = (org.springframework.security.core.userdetails.User) SecurityContextHolder.getContext().getAuthentication().getPrincipal();
        User user = userService.getUser(principal.getUsername());

        return ApiResponse.success("user", user);
    }

    @GetMapping("/access")
    public ApiResponse accessToken () {

        Date now = new Date();
        AuthToken newAccessToken = tokenProvider.createAuthToken(
                MASTER_ID,
                ROLE_USER,
                new Date(now.getTime() + (appProperties.getAuth().getTokenExpiry()*10000))
        );

        return ApiResponse.success("token", newAccessToken.getToken());
    }
}
