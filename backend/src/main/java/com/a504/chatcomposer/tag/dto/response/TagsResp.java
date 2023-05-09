package com.a504.chatcomposer.tag.dto.response;

import java.util.List;

import com.a504.chatcomposer.tag.dto.Tag;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Schema(description = "태그 목록 조회 응답 DTO")
@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
@JsonNaming(value = PropertyNamingStrategies.SnakeCaseStrategy.class)
public class TagsResp {

	@Schema(description = "태그 정보")
	@JsonProperty("tags")
	public List<Tag> tags;
}
