package com.a504.chatcomposer.tag.controller;

import java.util.List;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.a504.chatcomposer.global.util.BaseResponseBody;
import com.a504.chatcomposer.tag.dto.request.TagReq;
import com.a504.chatcomposer.tag.dto.response.TagsResp;
import com.a504.chatcomposer.tag.service.TagService;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;

@Tag(name = "TagController", description = "TagController API Document")
@RestController
@RequestMapping("/tags")
@RequiredArgsConstructor
public class TagController {

	private final TagService tagService;

	@Operation(summary = "태그 생성", description = "사용자가 입력한 태그를 생성합니다.")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "태그 생성 완료"),
		@ApiResponse(responseCode = "404", description = "태그 생성 실패")
	})
	@PostMapping
	public ResponseEntity<?> createTag(@RequestBody TagReq tagReq) {

		tagService.createTag(tagReq.getTagName());
		return ResponseEntity.ok().body(new BaseResponseBody(200, "태그가 생성되었습니다"));
	}

	@Operation(summary = "태그 목록 조회", description = "모든 태그의 리스트를 조회합니다.")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "태그 목록 조회 성공"),
		@ApiResponse(responseCode = "404", description = "태그 목록 조회 실패")
	})
	@GetMapping
	public ResponseEntity<?> getTags() {

		List<com.a504.chatcomposer.tag.dto.Tag> tags = tagService.getTags();
		return ResponseEntity.ok().body(new TagsResp(tags));
	}
}
