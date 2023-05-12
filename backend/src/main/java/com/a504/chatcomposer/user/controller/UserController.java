package com.a504.chatcomposer.user.controller;

import com.a504.chatcomposer.global.common.ApiResponse;
import com.a504.chatcomposer.global.config.properties.AppProperties;
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
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Date;
import java.util.HashMap;
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
        User user = userService.getUser(principal.getUsername());
        //TODO : 좋아하는 장르 목록의 List를 선언하고 받아온다.
            //TODO : Genre 조회하는 Genre 서비스를 생성한 후 사용자 seq로 조회하는 메소드 구현해야함.
        //TODO : Map을 선언해서 거기에 넣어준다.
        //TODO : map을 반환한다.
        Map<String, Object> resultMap = new HashMap<>();
        resultMap.put("msg","회원정보를 조회하였습니다.");
        resultMap.put("result",user);


        return ResponseEntity.ok().body(resultMap); // TODO : 여기 아직 NULL 들어있다?!
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
