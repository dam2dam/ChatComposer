package com.a504.chatcomposer.music.dto;

import java.time.LocalDateTime;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;

@Schema(description = "프롬프트 정보 DTO")
@Getter
@Setter
@ToString
@NoArgsConstructor
@AllArgsConstructor
@JsonNaming(value = PropertyNamingStrategies.SnakeCaseStrategy.class)
public class PromptDetails {

	@Schema(description = "사용자의 요청")
	private String requestDescription;

	@Schema(description = "chatGPT의 응답")
	private String responseDescription;

	@Schema(description = "요청 시각", example = "2023-05-16T13:00:00")
	@JsonFormat(shape = JsonFormat.Shape.STRING, timezone = "Asia/Seoul")
	private LocalDateTime transferDate;
}
