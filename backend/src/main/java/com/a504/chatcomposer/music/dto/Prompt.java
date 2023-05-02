package com.a504.chatcomposer.music.dto;

import java.time.LocalDateTime;

import com.fasterxml.jackson.annotation.JsonProperty;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Prompt {

	@JsonProperty(value = "prompt_id")
	private Long id;

	@JsonProperty(value = "request_description")
	private String requestDescription;

	@JsonProperty(value = "response_description")
	private String responseDescription;

	@JsonProperty(value = "transfer_date")
	private LocalDateTime transferDate;
}
