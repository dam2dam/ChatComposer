package com.a504.chatcomposer.music.dto;

import java.time.LocalDateTime;

import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Schema(description = "프롬프트 정보 DTO")
@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
@JsonNaming(value = PropertyNamingStrategies.SnakeCaseStrategy.class)
public class Prompt {

	@Schema(description = "프롬프트 pk")
	private Long promptId;

	@Schema(description = "사용자의 요청")
	private String requestDescription;

	@Schema(description = "chatGPT의 응답")
	private String responseDescription;

	@Schema(description = "요청 시각")
	private LocalDateTime transferDate;
}
