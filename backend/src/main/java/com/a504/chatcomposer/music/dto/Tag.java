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
public class Tag {

	@JsonProperty(value = "tag_id")
	private Long id;

	@JsonProperty(value = "tag_name")
	private String tagName;
}
