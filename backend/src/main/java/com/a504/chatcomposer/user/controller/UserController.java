package com.a504.chatcomposer.user.controller;

import com.a504.chatcomposer.global.common.ApiResponse;
import com.a504.chatcomposer.global.config.properties.AppProperties;
import com.a504.chatcomposer.music.dto.enums.Genre;
import com.a504.chatcomposer.user.dto.request.UserUpdateReq;
import com.a504.chatcomposer.user.dto.response.UserResp;
import com.a504.chatcomposer.user.entity.User;
import com.a504.chatcomposer.user.service.UserService;
import com.a504.chatcomposer.oauth.token.AuthToken;
import com.a504.chatcomposer.oauth.token.AuthTokenProvider;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;

import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Tag(name = "UserController", description = "사용자 API Document")
@RestController
@RequestMapping("/users")
@RequiredArgsConstructor
public class UserController {

    private final String ROLE_USER = "ROLE_USER";
    private final AuthTokenProvider tokenProvider;
    private final UserService userService;
    private final AppProperties appProperties;
    @Value("${jwt.master}")
    private String MASTER_ID;

    @Operation(summary = "유저 정보 조회", description = "토큰에 담겨있는 사용자 고유 번호로 사용자를 조회합니다.")
    @ApiResponses(value = {
            @io.swagger.v3.oas.annotations.responses.ApiResponse(responseCode = "200", description = "사용자 조회 완료"),
            @io.swagger.v3.oas.annotations.responses.ApiResponse(responseCode = "404", description = "사용자 조회 실패")
    })
    @GetMapping
    public ResponseEntity<?> getUser() {
        // principal에서 토큰 정보를 받아올 수 있음!
        org.springframework.security.core.userdetails.User principal = (org.springframework.security.core.userdetails.User) SecurityContextHolder.getContext().getAuthentication().getPrincipal();
        String userId = principal.getUsername();

        UserResp userResp = userService.getUserResp(userId);
        Map<String, Object> resultMap = new HashMap<>();
        resultMap.put("msg","회원정보를 조회하였습니다.");
        resultMap.put("result",userResp);

        return ResponseEntity.ok().body(resultMap);
    }


    //
    @Operation(summary = "유저 정보 수정", description = "사용자의 닉네임, 선호 장르 목록을 수정합니다.")
    @ApiResponses(value = {
            @io.swagger.v3.oas.annotations.responses.ApiResponse(responseCode = "200", description = "사용자 조회 완료"),
            @io.swagger.v3.oas.annotations.responses.ApiResponse(responseCode = "404", description = "사용자 조회 실패")
    })
    @PatchMapping
    public ResponseEntity<?> updateUser(@RequestBody UserUpdateReq userUpdateReq) {
        // principal에서 토큰 정보를 받아올 수 있음!
        org.springframework.security.core.userdetails.User principal = (org.springframework.security.core.userdetails.User) SecurityContextHolder.getContext().getAuthentication().getPrincipal();
        String userId = principal.getUsername();
        userService.updateUser(userId, userUpdateReq);

        Map<String, Object> resultMap = new HashMap<>();
        resultMap.put("msg","회원정보를 수정하였습니다.");
        return ResponseEntity.ok().body(resultMap);
    }

    @Operation(summary = "엑세스 토큰 발급", description = "유효기간이 매우 긴 마스터 계정의 엑세스 토큰을 발급합니다.")
    @ApiResponses(value = {
            @io.swagger.v3.oas.annotations.responses.ApiResponse(responseCode = "200", description = "토큰 발급 완료"),
            @io.swagger.v3.oas.annotations.responses.ApiResponse(responseCode = "404", description = "토큰 발급 실패")
    })
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