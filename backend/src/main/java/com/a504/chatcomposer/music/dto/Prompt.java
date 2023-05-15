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

@Schema(description = "프롬프트 정보 DTO + 프롬프트 pk")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@JsonNaming(value = PropertyNamingStrategies.SnakeCaseStrategy.class)
public class Prompt extends PromptDetails {

	@Schema(description = "프롬프트 pk")
	private Long promptId;

	@Builder
	public Prompt(String requestDescription, String responseDescription, LocalDateTime transferDate,
		Long promptId) {
		super(requestDescription, responseDescription, transferDate);
		this.promptId = promptId;
	}
}
