package com.a504.chatcomposer.music.entity;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import javax.persistence.JoinColumn;
import javax.persistence.ManyToOne;
import javax.persistence.Table;

import com.a504.chatcomposer.tag.entity.Tag;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;

@Builder
@AllArgsConstructor
@NoArgsConstructor
@Getter
@Setter
@ToString
@Entity
@Table(name = "music_tag")
public class MusicTag {

	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	@Column(name = "music_tag_id", nullable = false)
	private Long id;

	@ManyToOne
	@JoinColumn(name = "music_id")
	private Music music;

	@ManyToOne
	@JoinColumn(name = "tag_id")
	private Tag tag;

	/* 연관관계 메서드 설정 */
	public void setMusic(Music music) {
		this.music = music;
		music.getMusicTags().add(this);
	}

	/* 연관관계 메서드 설정 */
	public void setTag(Tag tag) {
		this.tag = tag;
		tag.getMusicTags().add(this);
	}
}
