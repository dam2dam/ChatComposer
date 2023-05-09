package com.a504.chatcomposer.tag.repository;

import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;

import com.a504.chatcomposer.tag.entity.Tag;

public interface TagRepository extends JpaRepository<Tag, Long> {

	Optional<Tag> findByTagNameLike(String tagName);
}
