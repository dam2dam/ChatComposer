package com.a504.chatcomposer.music.dto;

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
public class Member {

	@JsonProperty(value = "member_id")
	private long memberId;

	@JsonProperty(value = "nickname")
	private String nickname;
}
