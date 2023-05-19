package com.a504.chatcomposer.tag.service;

import java.util.List;
import java.util.stream.Collectors;

import org.springframework.stereotype.Service;

import com.a504.chatcomposer.global.exception.CustomException;
import com.a504.chatcomposer.global.exception.CustomExceptionType;
import com.a504.chatcomposer.tag.entity.Tag;
import com.a504.chatcomposer.tag.repository.TagRepository;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Service
@RequiredArgsConstructor
@Slf4j
public class TagService {

	private final TagRepository tagRepository;

	/**
	 * 태그 생성
	 */
	public void createTag(String tagName) {

		Tag existTag = tagRepository.findByTagName(tagName).orElse(null);
		// 중복 태그 생성 예외 처리
		if (existTag != null) {
			throw new CustomException(CustomExceptionType.DUPLICATE_TAG);
		}
		// 태그 생성 후 저장
		Tag tag = Tag.builder()
			.tagName(tagName)
			.build();
		tagRepository.save(tag);

		log.info("TagService | createTag() tag : {}", tag);
	}

	/**
	 * 태그 목록 조회
	 */
	public List<com.a504.chatcomposer.tag.dto.Tag> getTags() {

		List<Tag> tags = tagRepository.findAll();
		// 태그 목록 조회 예외 처리
		// if (CollectionUtils.isEmpty(tags)) {
		// 	throw new CustomException(CustomExceptionType.TAG_NOT_FOUND);
		// }
		log.info("TagService | getTags() tags: {}", tags);

		return tags.stream().map(tag ->
				com.a504.chatcomposer.tag.dto.Tag.builder()
					.tagId(tag.getId())
					.tagName(tag.getTagName())
					.build())
			.collect(Collectors.toList());
	}
}
