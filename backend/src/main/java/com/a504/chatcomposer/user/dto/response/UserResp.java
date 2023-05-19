package com.a504.chatcomposer.user.dto.response;

import com.a504.chatcomposer.music.dto.enums.Genre;
import com.a504.chatcomposer.user.entity.User;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.*;

import java.util.List;

@Schema(description = "회원 정보 DTO")
@Getter
@Setter
@Builder
@ToString
@NoArgsConstructor
@AllArgsConstructor
@JsonNaming(value = PropertyNamingStrategies.SnakeCaseStrategy.class)
public class UserResp {

	@Schema(description = "사용자 pk")
	private String userId;

	@Schema(description = "사용자 번호")
	private Long memberId;

	@Schema(description = "닉네임")
	private String nickname;

	@Schema(description = "이메일")
	private String email;

	@Schema(description = "장르 목록")
	private List<Genre> favoriteGenres;

	public UserResp(User user, List<Genre> favoriteGenres) {
		this.userId = user.getUserId();
		this.memberId = user.getUserSeq();
		this.nickname = user.getNickname();
		this.email = user.getEmail();
		this.favoriteGenres = favoriteGenres;
	}
}